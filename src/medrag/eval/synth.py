"""Build a synthetic retrieval test set from the indexed chunks.

For sampled chunks, the local LLM writes a standalone question, a short answer and
an *exact evidence quote*. The quote (not the chunk id) is the gold label: any
retrieved chunk containing it counts as relevant. That keeps the set valid when
chunking changes in later phases.

The output contains verbatim textbook quotes, so it lives under data/ (gitignored).
"""

import json
import random
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from medrag.eval.common import content_tokens, normalize
from medrag.generate.llm import LLMClient, Message
from medrag.index.store import Chunk

# No boolean "skip" flag: small models fill JSON fields in order and tended to answer
# skip=true first and then write a perfectly good question anyway. Empty strings = skip.
SYNTH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["question", "answer", "evidence"],
}

SYNTH_SYSTEM = "You write exam questions used to test a medical textbook search engine."

SYNTH_PROMPT = """Passage:
\"\"\"
{text}
\"\"\"

Write ONE question that this passage answers.
- Ask about one specific medical fact (a disease, drug, mechanism, structure, value or criterion).
- The question must make sense to someone who has NOT seen the passage. Never mention \
"the passage", "the text", "this chapter", "the figure" or "the author".
- answer: the short answer, at most 15 words.
- evidence: copy, word for word, one sentence or phrase from the passage (8 to 40 words) that \
contains the answer.
- Only if the passage has no clear medical fact (for example an index, reference list or figure \
labels), return empty strings for all three fields."""

BANNED = re.compile(r"\b(passage|the text|this text|chapter|figure|fig\.|the author|according to)\b", re.I)


@dataclass(frozen=True)
class SynthItem:
    id: str
    question: str
    answer: str
    evidence: str
    source_chunk_id: str
    book: str
    subject: str


def usable_chunk(c: Chunk, min_tokens: int = 200) -> bool:
    """Skip tail fragments and figure-label/index debris."""
    if c.n_tokens < min_tokens:
        return False
    letters = sum(ch.isalpha() for ch in c.text)
    lines = [ln for ln in c.text.splitlines() if ln.strip()]
    avg_line = sum(len(ln) for ln in lines) / max(len(lines), 1)
    return letters / max(len(c.text), 1) > 0.6 and avg_line > 40


def validate(item: dict[str, Any], chunk: Chunk) -> str | None:
    """Return a rejection reason, or None if the generated item is usable."""
    q, ev = item["question"].strip(), item["evidence"].strip()
    if not q:
        return "skipped"
    if not q.endswith("?"):
        return "not a question"
    if not 6 <= len(q.split()) <= 40:
        return "question length"
    if BANNED.search(q):
        return "refers to the passage"
    if not 5 <= len(ev.split()) <= 60:
        return "evidence length"
    if normalize(ev) not in normalize(chunk.text):
        return "evidence not verbatim"
    if not set(content_tokens(q)) & set(content_tokens(ev)):
        return "question unrelated to evidence"
    return None


def sample_chunks(chunks: Sequence[Chunk], n: int, seed: int) -> list[Chunk]:
    """Round-robin across books so small books are represented, shuffled within each book."""
    rng = random.Random(seed)
    by_book: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if usable_chunk(c):
            by_book[c.book].append(c)
    for lst in by_book.values():
        rng.shuffle(lst)
    order: list[Chunk] = []
    books = sorted(by_book)
    while len(order) < n and any(by_book.values()):
        for b in books:
            if by_book[b] and len(order) < n:
                order.append(by_book[b].pop())
    return order


def build_synthetic(
    chunks: Sequence[Chunk],
    llm: LLMClient,
    n: int,
    seed: int = 13,
    on_progress: Callable[[int, int, str | None], None] | None = None,
) -> tuple[list[SynthItem], dict[str, int]]:
    """Generate up to `n` validated items, trying up to 2n chunks. Returns items and rejection counts."""
    items: list[SynthItem] = []
    rejected: dict[str, int] = defaultdict(int)
    seen: set[str] = set()
    for i, chunk in enumerate(sample_chunks(chunks, 2 * n, seed)):
        if len(items) >= n:
            break
        out = llm.generate(
            [Message("system", SYNTH_SYSTEM), Message("user", SYNTH_PROMPT.format(text=chunk.text))],
            json_schema=SYNTH_SCHEMA,
        )
        try:
            raw = json.loads(out.text)
            reason = validate(raw, chunk)
        except (json.JSONDecodeError, KeyError, TypeError):
            reason = "invalid json"
        if reason is None and normalize(raw["question"]) in seen:
            reason = "duplicate"
        if reason:
            rejected[reason] += 1
        else:
            seen.add(normalize(raw["question"]))
            items.append(
                SynthItem(
                    id=f"syn-{len(items):04d}",
                    question=raw["question"].strip(),
                    answer=raw["answer"].strip(),
                    evidence=raw["evidence"].strip(),
                    source_chunk_id=chunk.chunk_id,
                    book=chunk.book,
                    subject=chunk.subject,
                )
            )
        if on_progress:
            on_progress(i + 1, len(items), reason)
    return items, dict(rejected)


def to_rows(items: Sequence[SynthItem]) -> list[dict[str, Any]]:
    return [asdict(i) for i in items]
