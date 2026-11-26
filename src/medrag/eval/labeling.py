"""Human labelling of answer sentences, to calibrate the automatic NLI faithfulness check.

Sentences come from a run's generation.jsonl. Sampling is stratified on the NLI score
(half judged supported, half unsupported) so both kinds of NLI error get measured.
Labels are appended to data/eval/nli_labels.jsonl after every answer, so labelling can
be stopped and resumed. That file quotes the textbooks, so it stays under data/ (gitignored).
"""

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from medrag.eval.common import content_tokens, read_jsonl
from medrag.eval.nli import strip_citations, windows


@dataclass(frozen=True)
class Candidate:
    key: str
    question: str
    sentence: str  # citations stripped
    sources: list[str]  # chunk ids retrieved for the answer
    nli_score: float  # max entailment over passages when the run was scored


@dataclass(frozen=True)
class Label:
    key: str
    question: str
    sentence: str
    sources: list[str]
    nli_score_at_sampling: float
    supported: bool
    collection: str = "medrag_chunks"  # where `sources` chunk ids resolve


def candidates_from_run(run_dir: Path) -> list[Candidate]:
    path = run_dir / "generation.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: pick a run that included the generation suite.")
    out = []
    for rec in read_jsonl(path):
        for s in rec.get("sentences", []):
            sentence = strip_citations(s["text"])
            key = hashlib.sha1((sentence + "|" + ",".join(rec["sources"])).encode()).hexdigest()[:16]
            out.append(Candidate(key, rec["question"], sentence, rec["sources"], s["max_support"]))
    return out


def stratified_sample(
    cands: Sequence[Candidate], n: int, done: set[str], seed: int = 11, threshold: float = 0.5
) -> list[Candidate]:
    """About n/2 above and n/2 below the NLI threshold; tops up from the other side if one runs short."""
    rng = random.Random(seed)
    todo = [c for c in cands if c.key not in done]
    hi = [c for c in todo if c.nli_score >= threshold]
    lo = [c for c in todo if c.nli_score < threshold]
    rng.shuffle(hi)
    rng.shuffle(lo)
    take_hi = min(len(hi), max(n // 2, n - len(lo)))
    picked = hi[:take_hi] + lo[: n - take_hi]
    rng.shuffle(picked)
    return picked


def best_excerpt(sentence: str, passage: str) -> tuple[float, str]:
    """The ~120-word window of `passage` sharing most content words with `sentence`."""
    target = set(content_tokens(sentence))
    best = (0.0, "")
    for w in windows(passage):
        overlap = len(target & set(content_tokens(w))) / max(len(target), 1)
        if overlap > best[0] or not best[1]:
            best = (overlap, w)
    return best


def load_labels(path: Path) -> list[Label]:
    return [Label(**d) for d in read_jsonl(path)] if path.exists() else []


HELP = "y = supported   n = not supported   m = show full passages   s = skip   q = save and quit"


def label_loop(
    items: Sequence[Candidate],
    passages: Callable[[str], str],
    out_path: Path,
    ask: Callable[[str], str],
    echo: Callable[[str], None],
    collection: str = "medrag_chunks",
) -> int:
    """Interactive loop. `passages(chunk_id)` returns chunk text. Returns how many labels were saved."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, c in enumerate(items, 1):
        texts = [passages(cid) for cid in c.sources]
        ranked = sorted(
            ((best_excerpt(c.sentence, t), j) for j, t in enumerate(texts)), key=lambda x: -x[0][0]
        )
        echo(f"\n[{i}/{len(items)}] Question: {c.question}")
        echo(f"\nSENTENCE:\n  {c.sentence}\n")
        for (_, excerpt), j in ranked[:2]:
            echo(f"--- most relevant excerpt of passage [{j + 1}] ({c.sources[j]}):\n{excerpt}\n")
        while True:
            a = ask("Is every claim in the SENTENCE supported by the passages? [y/n/m/s/q] ").strip().lower()
            if a == "m":
                for j, t in enumerate(texts):
                    echo(f"--- passage [{j + 1}] ({c.sources[j]}):\n{t.strip()}\n")
                continue
            if a in {"y", "n", "s", "q"}:
                break
            echo(HELP)
        if a == "q":
            break
        if a == "s":
            continue
        label = Label(c.key, c.question, c.sentence, c.sources, c.nli_score, a == "y", collection)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")
        saved += 1
    return saved


def label_rows(labels: Sequence[Label]) -> list[dict[str, Any]]:
    return [asdict(lb) for lb in labels]
