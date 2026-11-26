"""Text cleaning and structure recovery for the textbook files.

Each rule below exists because of artifacts measured in this corpus (see docs/ROADMAP.md, Phase 3):
- ligatures and odd spaces (DSM-5, First Aid 2, Janeway, Ross; em-spaces in Robbins captions);
- line-break hyphenation "Statisti- cal" (8k+ in DSM-5);
- running page headers repeated many times (Harrison);
- figure *labels* on their own line ("Fig. 1.1"), while real captions are kept;
- list markers ("2.") alone on a line, separated from their item;
- paragraphs broken mid-sentence;
- a whole book stored as a few giant lines (Surgery: 722k-character line).

`parse_book` returns blocks (headings and paragraphs) that the structured chunker uses.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

_INVISIBLE = re.compile("[­​‌‍⁠﻿]")
_SPACES = re.compile(r"[ \t  -   　]+")
_HYPHEN_BREAK = re.compile(r"\b([A-Za-z]{2,})- ([a-z]{2,})\b")
_KEEP_SUSPENDED = {"and", "or", "to", "nor", "versus", "vs"}  # "pre- and postoperative"
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]+")
_FIG_LABEL = re.compile(r"^(fig\.?|figure|table|box|plate)\s*[\dA-Z][\w.\-–]*\s*$", re.I)
_LIST_MARKER = re.compile(r"^(\d{1,2}|[A-Za-z]|[ivxlIVXL]{1,5})[.)]$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'“])")
_CHAPTER = re.compile(r"^(chapter|section|part)\s+\d+\b", re.I)
# Print-layout residue glued into the text (2,195x in Surgery), e.g.
# "Brunicardi_Ch03_p0083-p0102.indd 10108/12/18 10:07 AM"
_LAYOUT = re.compile(
    r"\S+\.indd\s+\d+\s*\d{2}/\d{2}/\d{2}\s+\d{1,2}:\d{2}\s*[AP]M|This page intentionally left blank"
)
_BYLINE = re.compile(r"\b(MD|PhD|PharmD|MPH|DO|RN|FRCP|FACS)\b")


@dataclass(frozen=True)
class Block:
    kind: str  # "heading" | "paragraph"
    text: str
    level: int = 0  # headings: 1 = chapter-like, 2 = section


def normalize_unicode(text: str) -> str:
    """NFKC folds ligatures (ﬁ -> fi) and exotic spaces; then drop invisible characters."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = _LAYOUT.sub(" ", text)
    return "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n"))


def build_vocab(text: str) -> Counter[str]:
    return Counter(w.lower() for w in _WORD.findall(text))


def dehyphenate(line: str, vocab: Counter[str]) -> str:
    """Rejoin 'statisti- cal' -> 'statistical', guided by how the word appears elsewhere in the book."""

    def fix(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        if b in _KEEP_SUSPENDED:
            return m.group()
        joined = (a + b).lower()
        if vocab[joined] > 0:
            return a + b
        if vocab[f"{a}-{b}".lower()] > 0:
            return f"{a}-{b}"
        return a + b

    return _HYPHEN_BREAK.sub(fix, line)


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s.strip()]


def split_long(text: str, target_chars: int = 1200, max_chars: int = 3000) -> list[str]:
    """Break an oversized paragraph into ~target_chars pieces at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    pieces, cur = [], ""
    for s in split_sentences(text):
        if cur and len(cur) + len(s) + 1 > target_chars:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    if cur:
        pieces.append(cur)
    # Spans without sentence boundaries (tables, lists) still get cut, at word boundaries.
    out = []
    for p in pieces:
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, target_chars)
            cut = cut if cut > 0 else target_chars
            out.append(p[:cut])
            p = p[cut:].lstrip()
        out.append(p)
    return out


def is_heading(line: str, next_line: str | None) -> bool:
    words = line.split()
    if not 1 <= len(words) <= 12 or len(line) > 90:
        return False
    if line[-1] in ".,;:(-" or not (line[0].isupper() or line[0].isdigit()):
        return False
    if _FIG_LABEL.match(line) or line.lower().startswith(("fig", "table", "http")):
        return False
    # Table rows, author bylines and broken sentences are not headings.
    if line.count(",") >= 2 or any(c in line for c in "%~=<>") or _BYLINE.search(line):
        return False
    if line.count("(") != line.count(")"):
        return False
    letters = sum(c.isalpha() for c in line)
    if letters < 0.6 * len(line.replace(" ", "")):
        return False
    # A heading introduces prose: the next line must be a real paragraph.
    return next_line is not None and len(next_line) >= 150


def parse_book(raw: str) -> list[Block]:
    text = normalize_unicode(raw)
    vocab = build_vocab(text)
    lines = [dehyphenate(ln, vocab) for ln in text.split("\n") if ln.strip()]

    # Running headers/footers: short lines repeated 3+ times. Keep only the first occurrence.
    counts = Counter(ln for ln in lines if len(ln) < 100)
    seen: set[str] = set()
    kept: list[str] = []
    for ln in lines:
        if counts.get(ln, 0) >= 3:
            if ln in seen:
                continue
            seen.add(ln)
        if _FIG_LABEL.match(ln) or ln.isdigit() or (len(ln) < 3 and not _LIST_MARKER.match(ln)):
            continue  # figure labels, page numbers, stray characters (but keep "2." markers)
        kept.append(ln)

    # Merge lone list markers into the next line; join paragraphs broken mid-sentence.
    merged: list[str] = []
    pending_marker = ""
    for ln in kept:
        if _LIST_MARKER.match(ln):
            pending_marker = ln
            continue
        if pending_marker:
            ln = f"{pending_marker} {ln}"
            pending_marker = ""
        if merged and (ln[0].islower() or ln.startswith(")")) and merged[-1][-1] not in ".!?:":
            merged[-1] = f"{merged[-1]} {ln}"
        else:
            merged.append(ln)

    paragraphs = [p for ln in merged for p in split_long(ln)]
    blocks = []
    for i, p in enumerate(paragraphs):
        nxt = paragraphs[i + 1] if i + 1 < len(paragraphs) else None
        if is_heading(p, nxt):
            blocks.append(Block("heading", p, 1 if _CHAPTER.match(p) else 2))
        else:
            blocks.append(Block("paragraph", p))
    return blocks


def clean_text(blocks: list[Block]) -> str:
    """The cleaned book as text: one block per paragraph, blank line between blocks."""
    return "\n\n".join(b.text for b in blocks)
