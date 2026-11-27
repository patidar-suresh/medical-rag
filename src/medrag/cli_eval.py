"""`medrag eval ...` commands: build datasets, run suites, compare runs."""

import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from medrag.config import get_settings

if TYPE_CHECKING:
    from medrag.eval.suites import SuiteResult
    from medrag.rag import RAGPipeline

eval_app = typer.Typer(help="Evaluation harness: datasets, suites, scorecards.", no_args_is_help=True)


class Suite(StrEnum):
    all = "all"
    retrieval = "retrieval"
    generation = "generation"
    smoke = "smoke"
    medqa = "medqa"


def _synth_path() -> Path:
    return get_settings().data_dir / "eval" / "synthetic.jsonl"


def _medqa_path() -> Path:
    from medrag.eval.medqa import TEST_FILE

    return get_settings().data_dir / "medqa" / TEST_FILE


@eval_app.command("fetch-medqa")
def fetch_medqa_cmd() -> None:
    """Download the MedQA USMLE 4-option test set (CC-BY-4.0) into data/medqa/."""
    from medrag.eval.common import read_jsonl
    from medrag.eval.medqa import fetch_medqa

    path = fetch_medqa(_medqa_path().parent)
    typer.echo(f"Saved {len(read_jsonl(path))} questions to {path}")


@eval_app.command("build-synth")
def build_synth(
    n: Annotated[int, typer.Option(help="Number of questions to keep.")] = 300,
    seed: Annotated[int, typer.Option(help="Sampling seed (fixes which chunks are used).")] = 13,
    model: Annotated[
        str | None, typer.Option(help="Ollama model to write questions (default: config).")
    ] = None,
) -> None:
    """Generate the synthetic retrieval set (question + evidence quote) from indexed chunks."""
    from medrag.eval.common import load_chunks, write_jsonl
    from medrag.eval.synth import build_synthetic, to_rows
    from medrag.generate.llm import OllamaClient

    s = get_settings()
    if model:
        s = s.model_copy(update={"llm_model": model})
    chunks = load_chunks(s.chunks_path)
    t0 = time.perf_counter()

    def progress(tried: int, kept: int, reason: str | None) -> None:
        if tried % 25 == 0:
            typer.echo(f"  tried {tried}, kept {kept} ({time.perf_counter() - t0:.0f}s)")

    items, rejected = build_synthetic(chunks, OllamaClient(s), n, seed, progress)
    write_jsonl(_synth_path(), to_rows(items))
    typer.echo(f"Kept {len(items)} questions -> {_synth_path()}  (model {s.llm_model})")
    typer.echo(f"Rejected: {rejected}")


@eval_app.command("run")
def run(
    suite: Annotated[list[Suite], typer.Option("--suite", "-s", help="Suite(s) to run.")] = [Suite.all],  # noqa: B006
    model: Annotated[
        list[str] | None, typer.Option("--model", "-m", help="Ollama model(s) to compare (repeatable).")
    ] = None,
    retrieval_limit: Annotated[
        int | None, typer.Option(help="Max synthetic questions for retrieval.")
    ] = None,
    gen_limit: Annotated[
        int, typer.Option(help="Synthetic questions answered in the generation suite.")
    ] = 40,
    medqa_limit: Annotated[int, typer.Option(help="MedQA questions (seeded sample).")] = 100,
) -> None:
    """Run evaluation suites and write a scorecard to eval/runs/<timestamp>-<model>/."""
    from medrag.cli import build_pipeline
    from medrag.eval.common import load_chunks, read_jsonl
    from medrag.eval.medqa import load_medqa
    from medrag.eval.nli import NLIScorer
    from medrag.eval.runner import Run
    from medrag.eval.smoke import load_cases
    from medrag.eval.suites import run_generation, run_medqa, run_retrieval, run_smoke
    from medrag.generate.llm import OllamaClient
    from medrag.index.embedder import SentenceTransformerEmbedder

    base = get_settings()
    suites = (
        {Suite.retrieval, Suite.generation, Suite.smoke, Suite.medqa} if Suite.all in suite else set(suite)
    )
    models = model or [base.llm_model]

    # Check inputs before spending time on models.
    synth = read_jsonl(_synth_path()) if suites & {Suite.retrieval, Suite.generation} else []
    if suites & {Suite.retrieval, Suite.generation} and not synth:
        raise typer.BadParameter("No synthetic set. Run `medrag eval build-synth` first.")
    medqa = load_medqa(_medqa_path(), medqa_limit) if Suite.medqa in suites else []
    installed = OllamaClient(base).installed_models()
    missing = [m for m in models if m not in installed]
    if missing:
        raise typer.BadParameter(f"Not pulled in Ollama: {missing}. Run: ollama pull <model>")

    embedder = SentenceTransformerEmbedder(base)
    chunks = load_chunks(base.chunks_path) if Suite.retrieval in suites else []
    nli: list[NLIScorer] = []  # loaded once, lazily

    def nli_factory() -> NLIScorer:
        if not nli:
            nli.append(NLIScorer(base.nli_model, base.embed_device))
        return nli[0]

    options = {
        "suites": sorted(x.value for x in suites),
        "retrieval_limit": retrieval_limit,
        "gen_limit": gen_limit,
        "medqa_limit": medqa_limit,
    }

    def run_suite(name: Suite, rag: "RAGPipeline", llm: OllamaClient) -> "SuiteResult":
        if name is Suite.retrieval:
            return run_retrieval(rag, synth[:retrieval_limit], chunks, typer.echo)
        if name is Suite.smoke:
            return run_smoke(rag, load_cases(Path("eval/smoke.jsonl")), typer.echo)
        if name is Suite.medqa:
            return run_medqa(rag, medqa, typer.echo)
        # Generation last: it unloads the LLM to make GPU room for the NLI model.
        return run_generation(rag, synth[:gen_limit], nli_factory, typer.echo, llm.unload, base.nli_threshold)

    for m in models:
        s = base.model_copy(update={"llm_model": m})
        rag = build_pipeline(s, embedder)
        llm = OllamaClient(s)
        run_ = Run(s, options)
        typer.echo(f"\n=== {m} -> {run_.dir}")
        for name in (Suite.retrieval, Suite.smoke, Suite.medqa, Suite.generation):
            if name in suites:
                typer.echo(f"[{name.value}]")
                t0 = time.perf_counter()
                run_.add(name.value, run_suite(name, rag, llm), time.perf_counter() - t0)
        llm.unload()  # free the GPU before the next model loads
        typer.echo(f"Report: {run_.dir / 'report.md'}")

    compare_cmd(last=len(models))


@eval_app.command("compare")
def compare_cmd(last: Annotated[int, typer.Option(help="How many recent runs to show.")] = 10) -> None:
    """Show key metrics of recent runs side by side."""
    from medrag.eval.runner import compare

    rows = compare(last=last)
    if len(rows) == 1:
        typer.echo("No runs yet. Run `medrag eval run`.")
        return
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        typer.echo("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)))


@eval_app.command("paired")
def paired_cmd(
    run_a: Annotated[str, typer.Argument(help="Baseline run (directory or name prefix).")],
    run_b: Annotated[str, typer.Argument(help="Candidate run (directory or name prefix).")],
    metric: Annotated[
        list[str] | None, typer.Option("--metric", "-m", help="Repeatable. Default: hit@1/3/5/10, medqa.")
    ] = None,
) -> None:
    """Is B really better than A? Exact McNemar test on the items both runs scored."""
    from medrag.eval.runner import PAIRED_METRICS, paired, resolve_run

    try:
        a, b = resolve_run(run_a), resolve_run(run_b)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    typer.echo(f"A = {a.name}\nB = {b.name}")
    for m in metric or list(PAIRED_METRICS):
        if m not in PAIRED_METRICS:
            raise typer.BadParameter(f"Unknown metric '{m}'. Choose from {', '.join(PAIRED_METRICS)}.")
        try:
            r = paired(a, b, m)
        except (FileNotFoundError, ValueError):
            continue  # suite not in both runs
        typer.echo(
            f"  {m:<18} n={r.n:<4} A {r.rate_a:.3f}  B {r.rate_b:.3f}  "
            f"gained {r.gained:>3} / lost {r.lost:>3}  p={r.p_value:.4f}"
        )


def _labels_path() -> Path:
    return get_settings().data_dir / "eval" / "nli_labels.jsonl"


def _latest_generation_run() -> Path:
    from medrag.eval.runner import RUNS_DIR

    runs = sorted(p.parent for p in RUNS_DIR.glob("*/generation.jsonl"))
    if not runs:
        raise typer.BadParameter("No run with a generation suite yet. Run `medrag eval run -s generation`.")
    return runs[-1]


def _run_collection(run_path: Path) -> str:
    import json

    cfg = json.loads((run_path / "config.json").read_text())
    return str(cfg["settings"]["collection"])


def _chunk_texts(collection: str) -> dict[str, str]:
    """chunk id -> text for a collection, including parent ids (parent-child collections)."""
    from medrag.eval.common import load_chunks

    path = get_settings().model_copy(update={"collection": collection}).chunks_path
    texts: dict[str, str] = {}
    for c in load_chunks(path):
        texts[c.chunk_id] = c.text
        if c.parent_id and c.parent_text:
            texts[c.parent_id] = c.parent_text
    return texts


@eval_app.command("label")
def label(
    n: Annotated[int, typer.Option(help="Sentences to label this session.")] = 50,
    run_dir: Annotated[
        Path | None, typer.Option("--run", help="Run directory to take sentences from (default: latest).")
    ] = None,
) -> None:
    """Label answer sentences as supported / not supported (saved after each answer; resumable)."""
    from medrag.eval.labeling import HELP, candidates_from_run, label_loop, load_labels, stratified_sample

    run_path = run_dir or _latest_generation_run()
    done = {lb.key for lb in load_labels(_labels_path())}
    items = stratified_sample(candidates_from_run(run_path), n, done, threshold=get_settings().nli_threshold)
    if not items:
        typer.echo(
            f"Nothing left to label in {run_path} ({len(done)} labels saved). Use --run for another run."
        )
        return
    typer.echo(f"Labelling {len(items)} sentences from {run_path}. {len(done)} labels already saved.")
    typer.echo("Judge only the passages shown, not your own medical knowledge.\n" + HELP)
    collection = _run_collection(run_path)
    chunks = _chunk_texts(collection)
    saved = label_loop(items, chunks.__getitem__, _labels_path(), input, typer.echo, collection)
    typer.echo(f"\nSaved {saved} labels ({len(done) + saved} total) -> {_labels_path()}")
    typer.echo("Next: medrag eval calibrate")


@eval_app.command("calibrate")
def calibrate(
    nli_model: Annotated[
        list[str] | None, typer.Option("--nli-model", help="NLI model(s) to compare (default: config).")
    ] = None,
) -> None:
    """Compare automatic support scores with your labels and recommend an NLI threshold."""
    from medrag.eval.calibration import best_threshold, confusion, lexical_support
    from medrag.eval.labeling import load_labels
    from medrag.eval.nli import NLIScorer

    s = get_settings()
    labels = load_labels(_labels_path())
    if len(labels) < 10:
        raise typer.BadParameter(f"Only {len(labels)} labels; label at least 10 (aim for 50) first.")
    truth = [lb.supported for lb in labels]
    by_collection = {c: _chunk_texts(c) for c in {lb.collection for lb in labels}}
    passages = [[by_collection[lb.collection][c] for c in lb.sources] for lb in labels]
    pos = sum(truth)
    typer.echo(f"{len(labels)} labels: {pos} supported, {len(labels) - pos} not supported\n")

    lex = [lexical_support(lb.sentence, p) for lb, p in zip(labels, passages, strict=True)]
    rows: list[tuple[str, float, float, float, float, float]] = []

    def add(name: str, scores: list[float], threshold: float) -> None:
        c = confusion(scores, truth, threshold)
        rows.append((name, threshold, c.balanced_accuracy, c.accuracy, c.precision, c.recall))

    add("lexical >= 0.9", lex, 0.9)
    for model in nli_model or [s.nli_model]:
        typer.echo(f"Scoring with {model} ...")
        scorer = NLIScorer(model, s.embed_device)
        nli = [
            max(scorer.support_matrix([lb.sentence], p)[0], default=0.0)
            for lb, p in zip(labels, passages, strict=True)
        ]
        short = model.split("/")[-1]
        add(f"{short} @ current", nli, s.nli_threshold)
        t, _ = best_threshold(nli, truth)
        add(f"{short} @ best", nli, t)
        both = [max(a, 1.0 if b >= 0.9 else 0.0) for a, b in zip(nli, lex, strict=True)]
        add(f"{short} OR lexical", both, t)
        del scorer

    typer.echo(f"\n{'scorer':<34}{'thresh':>7}{'bal.acc':>9}{'acc':>7}{'prec':>7}{'recall':>8}")
    for name, t, ba, acc, p, r in rows:
        typer.echo(f"{name:<34}{t:>7.2f}{ba:>9.3f}{acc:>7.3f}{p:>7.3f}{r:>8.3f}")
    best = max(rows, key=lambda r: r[2])
    typer.echo(f"\nBest agreement: {best[0]} (balanced accuracy {best[2]:.3f}).")
    typer.echo("precision = of sentences the scorer accepts, share you also accept; recall = of sentences")
    typer.echo(
        "you accept, share the scorer accepts. To adopt a threshold, set MEDRAG_NLI_THRESHOLD in .env."
    )
