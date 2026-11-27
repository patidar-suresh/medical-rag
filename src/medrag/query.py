"""Query preprocessing: medical abbreviation expansion from a curated dictionary."""

import re
from functools import lru_cache
from importlib import resources


@lru_cache
def abbreviations() -> dict[str, str]:
    text = resources.files("medrag").joinpath("resources/abbreviations.tsv").read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            abbr, expansion = line.split("\t", 1)
            out[abbr.strip()] = expansion.strip()
    return out


@lru_cache
def _pattern() -> re.Pattern[str]:
    words = sorted(abbreviations(), key=len, reverse=True)
    return re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, words)) + r")(?![\w-])")


def expand_abbreviations(query: str) -> str:
    """'Treatment of MI?' -> 'Treatment of MI (myocardial infarction)?'.

    The expansion is added next to the abbreviation, not substituted, so both spellings can
    match. Skipped when the query already spells the term out.
    """
    table = abbreviations()
    lowered = query.lower()

    def sub(m: re.Match[str]) -> str:
        full = table[m.group(1)]
        return m.group(1) if full.lower() in lowered else f"{m.group(1)} ({full})"

    return _pattern().sub(sub, query)
