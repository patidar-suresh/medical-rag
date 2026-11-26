"""Evaluation suites. Each returns a summary dict (metrics) and per-item records (for debugging)."""

import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from medrag.eval.common import (
    compact,
    latency_summary,
    mean,
    percentile,
    token_recall,
    wilson_interval,
)
from medrag.eval.medqa import MedQAItem, ask
from medrag.eval.metrics import first_relevant_rank, hit_rate, mrr, ndcg_at_k
from medrag.eval.nli import NLIScorer
from medrag.eval.smoke import SmokeCase, abstained, grade
from medrag.index.store import Chunk
from medrag.rag import Answer, RAGPipeline

Log = Callable[[str], None]


@dataclass
class SuiteResult:
    summary: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


# --------------------------------------------------------------------------- retrieval


def run_retrieval(
    rag: RAGPipeline,
    items: Sequence[dict[str, Any]],
    chunks: Sequence[Chunk],
    log: Log,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> SuiteResult:
    """Gold = evidence quote. A returned passage is relevant if it contains the quote.

    "Returned passage" = what the LLM would see: the parent section in parent-child mode.
    Larger passages make hits easier, so `context_tokens@5` is reported alongside.
    """
    kmax = max(ks)
    returned = {c.as_parent().chunk_id: c.as_parent().text for c in chunks}  # dedupes parents
    corpus = [compact(t) for t in returned.values()]
    ranks: list[int | None] = []
    ndcgs: list[float] = []
    book_hits: list[bool] = []
    context_tokens: list[float] = []
    latencies: list[float] = []
    by_subject: dict[str, list[int | None]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    skipped: list[str] = []

    for n, it in enumerate(items, 1):
        ev = compact(it["evidence"])
        n_relevant = sum(ev in t for t in corpus)
        if n_relevant == 0:  # quote no longer inside one passage (e.g. split by new chunking)
            skipped.append(it["id"])
            continue
        t0 = time.perf_counter()
        hits = rag.retrieve(it["question"], kmax)
        latencies.append(time.perf_counter() - t0)
        rel = [ev in compact(h.chunk.text) for h in hits]
        rank = first_relevant_rank(rel)
        ranks.append(rank)
        ndcgs.append(ndcg_at_k(rel, n_relevant, 10))
        book_hits.append(any(h.chunk.book == it["book"] for h in hits[:5]))
        context_tokens.append(float(sum(h.chunk.n_tokens for h in hits[:5])))
        by_subject[it["subject"]].append(rank)
        records.append(
            {
                "id": it["id"],
                "question": it["question"],
                "gold_book": it["book"],
                "rank": rank,
                "n_relevant_in_corpus": n_relevant,
                "retrieved": [h.chunk.chunk_id for h in hits],
                "scores": [round(h.score, 4) for h in hits],
            }
        )
        if n % 50 == 0:
            log(f"  retrieval {n}/{len(items)}  hit@5 so far {hit_rate(ranks, 5):.3f}")

    summary: dict[str, Any] = {"n": len(ranks), "skipped_no_gold": len(skipped)}
    summary.update({f"hit@{k}": _r(hit_rate(ranks, k)) for k in ks})
    summary["mrr@10"] = _r(mrr(ranks, 10))
    summary["ndcg@10"] = _r(mean(ndcgs))
    summary["book_hit@5"] = _r(mean([float(b) for b in book_hits]))
    summary["context_tokens@5"] = _r(mean(context_tokens), 1)
    summary["skipped_ids"] = skipped
    summary.update({k: _r(v) for k, v in latency_summary("retrieval", latencies).items()})
    summary["hit@5_by_subject"] = {s: _r(hit_rate(r, 5), 3) for s, r in sorted(by_subject.items())}
    return SuiteResult(summary, records)


# --------------------------------------------------------------------------- generation


def _answer_all(rag: RAGPipeline, questions: Sequence[str], log: Log, label: str) -> list[Answer]:
    answers = []
    for n, q in enumerate(questions, 1):
        answers.append(rag.answer(q))
        if n % 10 == 0 or n == len(questions):
            log(f"  {label} {n}/{len(questions)}")
    return answers


def _ops(answers: Sequence[Answer]) -> dict[str, Any]:
    c = [a.completion for a in answers]
    out: dict[str, Any] = {}
    out.update(latency_summary("retrieval", [a.seconds_retrieval for a in answers]))
    out.update(latency_summary("generation", [a.seconds_generation for a in answers]))
    ttft = [x.ttft_s for x in c if x.ttft_s is not None]
    out["ttft_p50_s"], out["ttft_p95_s"] = percentile(ttft, 50), percentile(ttft, 95)
    out["tokens_per_s"] = mean([x.tokens_per_s for x in c if x.tokens_per_s])
    out["prompt_tokens_mean"] = mean([float(x.prompt_tokens) for x in c if x.prompt_tokens])
    out["truncated"] = sum(x.truncated for x in c)
    return {k: _r(v, 3) if isinstance(v, float) else v for k, v in out.items()}


def run_generation(
    rag: RAGPipeline,
    items: Sequence[dict[str, Any]],
    nli_factory: Callable[[], NLIScorer],
    log: Log,
    before_nli: Callable[[], None] = lambda: None,
    threshold: float = 0.5,
) -> SuiteResult:
    """Answer synthetic questions, then score answers with NLI (faithfulness, citations)."""
    answers = _answer_all(rag, [it["question"] for it in items], log, "generation")
    before_nli()  # e.g. unload the LLM so the NLI model has GPU memory
    nli = nli_factory()

    faith: list[float] = []
    sentences = with_cite = cites = valid = supporting = 0
    recalls: list[float] = []
    refusals = 0
    records = []
    for it, a in zip(items, answers, strict=True):
        refused = abstained(a.text)
        refusals += refused
        rec: dict[str, Any] = {
            "id": it["id"],
            "question": it["question"],
            "reference_answer": it["answer"],
            "answer": a.text,
            "abstained": refused,
            "sources": [h.chunk.chunk_id for h in a.hits],
        }
        recall = token_recall(it["answer"], a.text)
        recalls.append(recall)
        rec["answer_recall"] = round(recall, 3)
        if not refused:
            g = nli.ground(a.text, [h.chunk.text for h in a.hits], threshold)
            if g.faithfulness is not None:
                faith.append(g.faithfulness)
            w, t, v, s = g.citation_counts()
            sentences += len(g.sentences)
            with_cite, cites, valid, supporting = with_cite + w, cites + t, valid + v, supporting + s
            rec["faithfulness"] = _r(g.faithfulness, 3)
            rec["sentences"] = [
                {"text": s.sentence, "max_support": round(max(s.support, default=0), 3), "cited": s.cited}
                for s in g.sentences
            ]
        records.append(rec)

    n = len(items)
    summary: dict[str, Any] = {
        "n": n,
        "nli_threshold": threshold,
        "faithfulness": _r(mean(faith)),
        "citation_coverage": _r(with_cite / sentences) if sentences else None,
        "citation_validity": _r(valid / cites) if cites else None,
        "citation_precision": _r(supporting / valid) if valid else None,
        "answer_recall": _r(mean(recalls)),
        "answer_recall>=0.5": _r(mean([float(r >= 0.5) for r in recalls])),
        "false_refusal_rate": _r(refusals / n) if n else None,
    }
    summary.update(_ops(answers))
    return SuiteResult(summary, records)


# --------------------------------------------------------------------------- smoke


def run_smoke(rag: RAGPipeline, cases: Sequence[SmokeCase], log: Log) -> SuiteResult:
    answers = _answer_all(rag, [c.question for c in cases], log, "smoke")
    results = [grade(c, a) for c, a in zip(cases, answers, strict=True)]
    answerable = [r for r in results if r.retrieval_ok is not None]
    unanswerable = [r for r in results if r.retrieval_ok is None]
    summary: dict[str, Any] = {
        "n": len(results),
        "pass_rate": _r(mean([float(r.answer_ok) for r in results])),
        "answerable_pass_rate": _r(mean([float(r.answer_ok) for r in answerable])),
        "retrieval_book_hit": _r(mean([float(bool(r.retrieval_ok)) for r in answerable])),
        "correct_refusal_rate": _r(mean([float(r.answer_ok) for r in unanswerable])),
        "false_refusal_rate": _r(mean([float(abstained(r.answer.text)) for r in answerable])),
    }
    summary.update(_ops(answers))
    records = [
        {
            "id": r.case.id,
            "question": r.case.question,
            "answer": r.answer.text,
            "answer_ok": r.answer_ok,
            "retrieval_ok": r.retrieval_ok,
            "sources": [h.chunk.chunk_id for h in r.answer.hits],
        }
        for r in results
    ]
    return SuiteResult(summary, records)


# --------------------------------------------------------------------------- medqa


def run_medqa(rag: RAGPipeline, items: Sequence[MedQAItem], log: Log) -> SuiteResult:
    """Each question twice: closed-book, then with the top-k retrieved passages."""
    closed_ok = rag_ok = closed_invalid = rag_invalid = helped = hurt = 0
    gen_s: list[float] = []
    records = []
    for n, it in enumerate(items, 1):
        c_letter, c_out = ask(rag.llm, it, None)
        hits = rag.retrieve(it.question)
        t0 = time.perf_counter()
        r_letter, r_out = ask(rag.llm, it, hits)
        gen_s.append(time.perf_counter() - t0)
        c_right, r_right = c_letter == it.answer, r_letter == it.answer
        closed_ok += c_right
        rag_ok += r_right
        closed_invalid += c_letter is None
        rag_invalid += r_letter is None
        helped += r_right and not c_right
        hurt += c_right and not r_right
        records.append(
            {
                "id": it.id,
                "step": it.step,
                "gold": it.answer,
                "closed_book": c_letter,
                "rag": r_letter,
                "sources": [h.chunk.chunk_id for h in hits],
            }
        )
        if n % 20 == 0 or n == len(items):
            log(f"  medqa {n}/{len(items)}  closed {closed_ok / n:.3f}  rag {rag_ok / n:.3f}")

    n = len(items)
    ci_c, ci_r = wilson_interval(closed_ok, n), wilson_interval(rag_ok, n)
    summary: dict[str, Any] = {
        "n": n,
        "acc_closed_book": _r(closed_ok / n) if n else None,
        "acc_closed_book_95ci": [_r(x, 3) for x in ci_c] if ci_c else None,
        "acc_rag": _r(rag_ok / n) if n else None,
        "acc_rag_95ci": [_r(x, 3) for x in ci_r] if ci_r else None,
        "rag_helped": helped,
        "rag_hurt": hurt,
        "invalid_closed_book": closed_invalid,
        "invalid_rag": rag_invalid,
    }
    summary.update({k: _r(v, 3) for k, v in latency_summary("rag_generation", gen_s).items()})
    return SuiteResult(summary, records)
