"""Shared helpers for the evaluation harness: JSONL IO, text normalisation, statistics."""

import json
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from medrag.index.store import Chunk


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    """Chunks written by `medrag ingest` (must match what is in Qdrant)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `medrag ingest` first.")
    return [Chunk(**d) for d in read_jsonl(path)]


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, keep only letters/digits separated by single spaces (for robust quote matching)."""
    return " " + _NON_ALNUM.sub(" ", text.lower()).strip() + " "


# fmt: off
_NON_ALNUM_ALL = re.compile(r"[^a-z0-9]+")


def compact(text: str) -> str:
    """Letters and digits only, no spaces, after Unicode folding.

    Used to match gold evidence quotes against chunks: robust to the cleaning step
    (ligatures, rejoined hyphenation "statisti- cal", collapsed whitespace).
    """
    return _NON_ALNUM_ALL.sub("", unicodedata.normalize("NFKC", text).lower())


STOPWORDS = frozenset((
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "which", "with",
    "what", "who", "whom", "whose", "when", "where", "why", "how", "most", "more", "than", "other",
    "also", "can", "may", "not", "no",
))
# fmt: on


def content_tokens(text: str) -> list[str]:
    return [t for t in normalize(text).split() if t not in STOPWORDS]


def token_recall(reference: str, candidate: str) -> float:
    """Fraction of the reference's content words that appear in the candidate (SQuAD-style recall)."""
    ref = content_tokens(reference)
    if not ref:
        return 0.0
    cand = set(content_tokens(candidate))
    return sum(t in cand for t in ref) / len(ref)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile, p in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% confidence interval for a proportion; honest for small samples, unlike +/- 1.96*SE."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def latency_summary(prefix: str, seconds: Sequence[float]) -> dict[str, float | None]:
    return {f"{prefix}_p50_s": percentile(seconds, 50), f"{prefix}_p95_s": percentile(seconds, 95)}
