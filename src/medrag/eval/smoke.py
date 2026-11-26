"""Smoke suite over hand-written cases in `eval/smoke.jsonl`.

Simple, transparent checks (keyword match, source-book hit, abstention on
unanswerable questions). Keyword matching is lenient by design; the generation
suite's NLI faithfulness and citation metrics are the stricter signal.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from medrag.generate.prompts import NOT_FOUND
from medrag.rag import Answer


@dataclass(frozen=True)
class SmokeCase:
    id: str
    question: str
    answerable: bool = True
    keywords: tuple[str, ...] = ()  # pass if ANY appears in the answer
    keywords_all: tuple[str, ...] = ()  # and ALL of these appear
    books: tuple[str, ...] = ()  # books expected to contain the answer


@dataclass(frozen=True)
class SmokeResult:
    case: SmokeCase
    answer: Answer
    answer_ok: bool
    retrieval_ok: bool | None  # None for unanswerable cases


def load_cases(path: Path) -> list[SmokeCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            for key in ("keywords", "keywords_all", "books"):
                d[key] = tuple(d.get(key, ()))
            cases.append(SmokeCase(**d))
    return cases


# Small models paraphrase the exact refusal sentence, so accept common variants.
ABSTAIN_MARKERS = (
    NOT_FOUND.lower().rstrip("."),
    "could not find",
    "couldn't find",
    "cannot find",
    "do not contain",
    "does not contain",
    "not mentioned in",
    "no information",
)


def abstained(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in ABSTAIN_MARKERS)


def grade(case: SmokeCase, answer: Answer) -> SmokeResult:
    text = answer.text.lower()
    abstained_ = abstained(text)
    if not case.answerable:
        return SmokeResult(case, answer, abstained_, None)
    any_ok = not case.keywords or any(k.lower() in text for k in case.keywords)
    all_ok = all(k.lower() in text for k in case.keywords_all)
    hit_books = {h.chunk.book for h in answer.hits}
    retrieval_ok = not case.books or bool(hit_books & set(case.books))
    return SmokeResult(case, answer, any_ok and all_ok and not abstained_, retrieval_ok)
