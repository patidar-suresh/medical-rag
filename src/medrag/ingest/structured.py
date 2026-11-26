"""Structure-aware chunking: pack whole paragraphs (or sentences) under their section headings.

Compared with fixed token windows (chunker.py), chunks
- start and end on paragraph or sentence boundaries, never mid-sentence;
- prefer to break at a heading, so a chunk tends to cover one topic;
- carry a breadcrumb ("Book › Chapter › Section") usable as a contextual header.

Units (headings, paragraphs, or sentences of long paragraphs) are measured in the
embedding model's tokens, using one batched tokenizer call per book.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from medrag.ingest.chunker import OffsetsFn, chunk_spans
from medrag.ingest.clean import Block

_SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+[\"')\]]*|$)\s*")


@dataclass(frozen=True)
class Unit:
    start: int  # char offsets into the cleaned book text
    end: int
    n_tokens: int
    is_heading: bool
    crumb: tuple[str, ...]  # (chapter, section) headings in force at this unit


@dataclass(frozen=True)
class Piece:
    start: int
    end: int
    n_tokens: int
    context: str  # "Book › Chapter › Section"
    units: tuple[int, int]  # [first, last) unit indices


def block_offsets(blocks: Sequence[Block]) -> list[int]:
    """Start of each block in clean_text(blocks) == '\\n\\n'.join(...)."""
    starts, pos = [], 0
    for b in blocks:
        starts.append(pos)
        pos += len(b.text) + 2
    return starts


def make_units(blocks: Sequence[Block], offsets_fn: OffsetsFn, max_tokens: int) -> list[Unit]:
    """One unit per heading/paragraph; paragraphs over `max_tokens` become sentences (or token windows)."""
    starts = block_offsets(blocks)
    token_offs = offsets_fn([b.text for b in blocks])
    units: list[Unit] = []
    chapter = section = ""
    for b, b0, toks in zip(blocks, starts, token_offs, strict=True):
        if b.kind == "heading":
            if b.level == 1:
                chapter, section = b.text, ""
            else:
                section = b.text
        crumb = tuple(x for x in (chapter, section) if x)
        if b.kind == "heading" or len(toks) <= max_tokens:
            units.append(Unit(b0, b0 + len(b.text), len(toks), b.kind == "heading", crumb))
            continue
        for m in _SENTENCE.finditer(b.text):
            s, e = m.start(), m.end() - (len(m.group()) - len(m.group().rstrip()))
            if e <= s:
                continue
            sent_toks = [(a, z) for a, z in toks if a >= s and z <= e]
            if len(sent_toks) <= max_tokens:
                units.append(Unit(b0 + s, b0 + e, len(sent_toks), False, crumb))
            else:  # a "sentence" longer than a chunk (tables, lists): fixed windows
                for sp in chunk_spans(sent_toks, max_tokens, 0):
                    units.append(Unit(b0 + sp.start, b0 + sp.end, sp.n_tokens, False, crumb))
    return units


def pack(units: Sequence[Unit], lo: int, hi: int, budget: int, overlap: int) -> list[tuple[int, int]]:
    """Greedily group units[lo:hi] into ranges of <= budget tokens.

    A heading starts a new chunk once the current one is at least a third full, and
    up to `overlap` tokens of trailing non-heading units are repeated at the start of
    the next chunk when a chunk is cut for size.
    """
    ranges: list[tuple[int, int]] = []
    start, tokens = lo, 0
    i = lo
    while i < hi:
        u = units[i]
        heading_break = u.is_heading and i > start and tokens >= budget // 3
        if i > start and (tokens + u.n_tokens > budget or heading_break):
            ranges.append((start, i))
            back, carried = i, 0
            if not heading_break:
                while back - 1 > start and not units[back - 1].is_heading:
                    if carried + units[back - 1].n_tokens > overlap:
                        break
                    back -= 1
                    carried += units[back].n_tokens
            start, tokens = back, carried
        tokens += u.n_tokens
        i += 1
    if start < hi:
        ranges.append((start, hi))
    return ranges


def breadcrumb(title: str, crumb: tuple[str, ...]) -> str:
    return " › ".join((title, *crumb))


def to_pieces(units: Sequence[Unit], ranges: Sequence[tuple[int, int]], title: str) -> list[Piece]:
    return [
        Piece(
            units[a].start,
            units[b - 1].end,
            sum(u.n_tokens for u in units[a:b]),
            breadcrumb(title, units[a].crumb),
            (a, b),
        )
        for a, b in ranges
    ]
