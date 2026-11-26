"""`medrag` command-line interface."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from medrag.config import Settings, get_settings
from medrag.doctor import run_checks
from medrag.logs import configure_logging

if TYPE_CHECKING:
    from medrag.rag import RAGPipeline

app = typer.Typer(help="MedRAG: local medical RAG over textbooks.", no_args_is_help=True)


@app.callback()
def main() -> None:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)


@app.command()
def doctor() -> None:
    """Check that Qdrant, Ollama (and the configured model), GPU and corpus are ready."""
    results = run_checks(get_settings())
    for r in results:
        mark = "OK  " if r.ok else ("FAIL" if r.required else "WARN")
        typer.echo(f"[{mark}] {r.name:<7} {r.detail}")
    if any(r.required and not r.ok for r in results):
        raise typer.Exit(code=1)


@app.command()
def ingest(
    books: Annotated[
        list[str] | None, typer.Option("--book", "-b", help="Only ingest this book (repeatable).")
    ] = None,
) -> None:
    """Chunk, embed and index the textbooks into Qdrant (rebuilds the collection)."""
    from medrag.generate.llm import OllamaClient
    from medrag.index.embedder import SentenceTransformerEmbedder
    from medrag.index.store import QdrantStore
    from medrag.ingest.chunker import hf_offsets_fn
    from medrag.ingest.loader import load_books
    from medrag.ingest.pipeline import ingest as run_ingest

    s = get_settings()
    loaded = load_books(s.corpus_dir, books)
    typer.echo(f"Loaded {len(loaded)} book(s) from {s.corpus_dir}")

    # The LLM and the embedder share one small GPU: free the LLM's memory first.
    try:
        OllamaClient(s).unload()
    except Exception:  # noqa: BLE001 - Ollama being down shouldn't block ingestion
        typer.echo("Note: could not unload the Ollama model (is Ollama running?)")

    embedder = SentenceTransformerEmbedder(s)
    typer.echo(f"Embedding with {s.embed_model} on {embedder.device}")
    stats = run_ingest(
        loaded,
        settings=s,
        offsets_fn=hf_offsets_fn(s.embed_model),
        embedder=embedder,
        store=QdrantStore.from_url(s.qdrant_url, s.collection),
        show_progress=True,
    )
    typer.echo(
        f"Indexed {stats.chunks} chunks ({stats.tokens:,} tokens) from {stats.books} book(s) "
        f"into '{s.collection}'. Time: chunking {stats.seconds_chunking:.0f}s, "
        f"embedding {stats.seconds_embedding:.0f}s, indexing {stats.seconds_indexing:.0f}s"
    )


def _pipeline(s: Settings) -> "RAGPipeline":
    # Heavy imports (torch) stay inside commands so `medrag doctor` starts instantly.
    from medrag.generate.llm import OllamaClient
    from medrag.index.embedder import SentenceTransformerEmbedder
    from medrag.index.store import QdrantStore
    from medrag.rag import RAGPipeline

    store = QdrantStore.from_url(s.qdrant_url, s.collection)
    if not store.exists():
        typer.echo(f"Collection '{s.collection}' not found in Qdrant. Run `medrag ingest` first.", err=True)
        raise typer.Exit(code=1)
    return RAGPipeline(SentenceTransformerEmbedder(s), store, OllamaClient(s), s.top_k)


@app.command()
def ask(
    question: str,
    k: Annotated[int | None, typer.Option("--k", help="Passages to retrieve.")] = None,
    show_context: Annotated[bool, typer.Option(help="Print the retrieved passages.")] = False,
) -> None:
    """Answer a question from the indexed textbooks (streams the answer)."""
    rag = _pipeline(get_settings())
    t0 = time.perf_counter()
    hits, tokens = rag.stream_answer(question, k)
    for i, h in enumerate(hits, 1):
        if show_context:
            typer.echo(f"--- [{i}] {h.chunk.chunk_id} (score {h.score:.3f})\n{h.chunk.text.strip()}\n")
    for t in tokens:
        typer.echo(t, nl=False)
    typer.echo("\n\nSources:")
    for i, h in enumerate(hits, 1):
        typer.echo(f"  [{i}] {h.chunk.chunk_id}  (score {h.score:.3f})")
    typer.echo(f"({time.perf_counter() - t0:.1f}s)")


@app.command()
def smoke(
    path: Annotated[Path, typer.Option(help="Smoke-test cases (JSONL).")] = Path("eval/smoke.jsonl"),
) -> None:
    """Run the smoke questions and report answer and retrieval pass rates."""
    from medrag.smoke import grade, load_cases

    s = get_settings()
    rag = _pipeline(s)
    cases = load_cases(path)
    results = []
    for c in cases:
        r = grade(c, rag.answer(c.question))
        results.append(r)
        ret = "-" if r.retrieval_ok is None else ("hit " if r.retrieval_ok else "miss")
        typer.echo(
            f"{'PASS' if r.answer_ok else 'FAIL'}  retrieval:{ret:<4}  "
            f"{r.answer.seconds_generation:5.1f}s  {c.id}: {r.answer.text[:90]!r}"
        )

    answerable = [r for r in results if r.retrieval_ok is not None]
    typer.echo(
        f"\nAnswers correct: {sum(r.answer_ok for r in results)}/{len(results)}   "
        f"Retrieval hit (answerable): {sum(bool(r.retrieval_ok) for r in answerable)}/{len(answerable)}   "
        f"model={s.llm_model} embed={s.embed_model} k={s.top_k} chunk={s.chunk_size}/{s.chunk_overlap}"
    )

    out = Path("eval/runs") / f"{datetime.now():%Y%m%d-%H%M%S}-smoke.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            rec = {
                "id": r.case.id,
                "question": r.case.question,
                "answer": r.answer.text,
                "answer_ok": r.answer_ok,
                "retrieval_ok": r.retrieval_ok,
                "sources": [h.chunk.chunk_id for h in r.answer.hits],
                "scores": [round(h.score, 4) for h in r.answer.hits],
                "seconds_generation": round(r.answer.seconds_generation, 2),
                "model": s.llm_model,
            }
            f.write(json.dumps(rec) + "\n")
    typer.echo(f"Details: {out}")


if __name__ == "__main__":
    app()
