"""Suites end to end with fakes: in-memory Qdrant, keyword embedder, scripted LLM, fake NLI."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fakes import EchoLLM, KeywordEmbedder
from qdrant_client import QdrantClient

from medrag.eval.medqa import MedQAItem, load_medqa
from medrag.eval.nli import AnswerGrounding, SentenceGrounding, split_sentences
from medrag.eval.runner import Run, compare
from medrag.eval.suites import run_generation, run_medqa, run_retrieval
from medrag.eval.synth import build_synthetic, sample_chunks, usable_chunk, validate
from medrag.generate.llm import Completion, Message
from medrag.index.store import Chunk, QdrantStore
from medrag.rag import RAGPipeline

SEIZURE = "Ethosuximide is the drug of choice for absence seizure in children. " * 20
NERVE = "The deltoid muscle is supplied by the axillary nerve from the posterior cord. " * 20


def chunk(cid: str, book: str, text: str, n_tokens: int = 300) -> Chunk:
    return Chunk(cid, book, "pharmacology", text, 0, len(text), n_tokens)


def make_rag(llm: Any = None) -> tuple[RAGPipeline, list[Chunk]]:
    chunks = [chunk("Pharm#0", "Pharm", SEIZURE), chunk("Anat#0", "Anat", NERVE)]
    emb = KeywordEmbedder()
    store = QdrantStore(QdrantClient(":memory:"), "t")
    store.recreate(emb.dim)
    store.upsert(chunks, emb.embed_documents([c.text for c in chunks]))
    return RAGPipeline(emb, store, llm or EchoLLM(), top_k=2), chunks


SYNTH = [
    {
        "id": "s1",
        "question": "What is the drug of choice for absence seizure?",
        "answer": "Ethosuximide",
        "evidence": "Ethosuximide is the drug of choice for absence seizure",
        "book": "Pharm",
        "subject": "pharmacology",
    },
    {
        "id": "s2",
        "question": "Which nerve supplies the deltoid?",
        "answer": "axillary nerve",
        "evidence": "supplied by the axillary nerve",
        "book": "Anat",
        "subject": "anatomy",
    },
]


def test_retrieval_suite() -> None:
    rag, chunks = make_rag()
    gone = {**SYNTH[0], "id": "s3", "evidence": "text that appears in no chunk at all"}
    res = run_retrieval(rag, [*SYNTH, gone], chunks, log=lambda _: None)
    assert res.summary["n"] == 2 and res.summary["skipped_no_gold"] == 1
    assert res.summary["hit@1"] == 1.0 and res.summary["mrr@10"] == 1.0
    assert res.summary["book_hit@5"] == 1.0


class FakeNLI:
    """Supports a sentence iff it mentions a word that appears in the passage it cites best."""

    def ground(self, answer: str, passages: list[str], threshold: float = 0.5) -> AnswerGrounding:
        from medrag.eval.nli import citations

        sents = []
        for s in split_sentences(answer):
            support = [
                0.9 if "ethosuximide" in s.lower() and "ethosuximide" in p.lower() else 0.1 for p in passages
            ]
            sents.append(SentenceGrounding(s, support, citations(s)))
        return AnswerGrounding(sents, threshold)


def test_generation_suite_unloads_llm_before_nli() -> None:
    calls: list[str] = []
    rag, _ = make_rag(EchoLLM("Ethosuximide is the drug of choice [1]. It also cures everything [2]."))
    res = run_generation(
        rag,
        SYNTH[:1],
        nli_factory=lambda: calls.append("nli") or FakeNLI(),  # type: ignore[func-returns-value,return-value]
        log=lambda _: None,
        before_nli=lambda: calls.append("unload"),
    )
    assert calls == ["unload", "nli"]
    s = res.summary
    assert s["faithfulness"] == 0.5 and s["citation_precision"] == 0.5 and s["citation_validity"] == 1.0
    assert s["answer_recall"] == 1.0 and s["false_refusal_rate"] == 0.0


class LetterLLM(EchoLLM):
    """Answers B without passages and A with passages."""

    def generate(self, messages: list[Message], *, json_schema: dict[str, Any] | None = None) -> Completion:
        letter = "A" if "Reference passages" in messages[-1].content else "B"
        return Completion(json.dumps({"answer": letter}), "fake", "stop", 10, 1)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        yield ""


def test_medqa_suite_and_loader(tmp_path: Path) -> None:
    p = tmp_path / "medqa.jsonl"
    rows = [
        {
            "question": f"Q{i} absence seizure?",
            "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
            "answer_idx": "A",
            "meta_info": "step1",
        }
        for i in range(5)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    items = load_medqa(p, limit=3)
    assert len(items) == 3 and isinstance(items[0], MedQAItem)
    rag, _ = make_rag(LetterLLM())
    s = run_medqa(rag, items, log=lambda _: None).summary
    assert (s["acc_closed_book"], s["acc_rag"], s["rag_helped"], s["rag_hurt"]) == (0.0, 1.0, 3, 0)


def test_synth_validation_and_sampling() -> None:
    c = chunk("Pharm#0", "Pharm", SEIZURE)
    ok = {
        "question": "What is the drug of choice for absence seizure in children?",
        "answer": "Ethosuximide",
        "evidence": "Ethosuximide is the drug of choice for absence seizure",
    }
    assert validate(ok, c) is None
    assert (
        validate({**ok, "question": "According to the passage, what is used?"}, c) == "refers to the passage"
    )
    assert validate({**ok, "evidence": "Valproate is the drug of choice for absence seizure"}, c) == (
        "evidence not verbatim"
    )
    assert validate({"question": "", "answer": "", "evidence": ""}, c) == "skipped"
    assert not usable_chunk(chunk("x#1", "x", "Fig. 1\n2\n3", n_tokens=20))
    many = [chunk(f"A#{i}", "A", SEIZURE) for i in range(5)] + [chunk("B#0", "B", NERVE)]
    picked = sample_chunks(many, 3, seed=1)
    assert {c.book for c in picked} == {"A", "B"}  # round-robin includes the small book


def test_build_synthetic_with_scripted_llm() -> None:
    reply = {
        "question": "What is the drug of choice for absence seizure?",
        "answer": "Ethosuximide",
        "evidence": "Ethosuximide is the drug of choice",
    }
    llm = EchoLLM(json.dumps(reply))
    items, rejected = build_synthetic([chunk(f"P#{i}", "P", SEIZURE) for i in range(4)], llm, n=2)
    assert len(items) == 1 and rejected == {"duplicate": 3}  # same question each time


def test_run_dir_and_compare(tmp_path: Path) -> None:
    from medrag.config import Settings
    from medrag.eval.suites import SuiteResult

    run = Run(Settings(_env_file=None), {"suites": ["retrieval"]}, root=tmp_path)
    run.add("retrieval", SuiteResult({"hit@5": 0.8, "mrr@10": 0.6}, [{"id": "x"}]), 1.0)
    assert {p.name for p in run.dir.iterdir()} == {
        "config.json",
        "retrieval.jsonl",
        "summary.json",
        "report.md",
    }
    rows = compare(tmp_path)
    assert rows[1][3] == "hybrid+rerank:ms-marco-MiniLM-L-6-v2" and rows[1][4] == "0.800"
    assert "| hit@5 | 0.800 |" in (run.dir / "report.md").read_text()
