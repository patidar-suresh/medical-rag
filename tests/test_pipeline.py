"""End-to-end ingest + answer with fakes and an in-memory Qdrant."""

import json
from pathlib import Path

import pytest
from fakes import EchoLLM, KeywordEmbedder, whitespace_offsets
from qdrant_client import QdrantClient

from medrag.config import Settings
from medrag.generate.prompts import NOT_FOUND
from medrag.index.store import QdrantStore
from medrag.ingest.loader import load_books, subject_for
from medrag.ingest.pipeline import ingest
from medrag.rag import RAGPipeline
from medrag.smoke import SmokeCase, abstained, grade


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "Pharmacology_Katzung.txt").write_text(
        "Absence seizure treatment: ethosuximide is first line for absence seizure.\n" * 3
    )
    (d / "Anatomy_Gray.txt").write_text("The deltoid is supplied by the axillary nerve.\n" * 3)
    return d


@pytest.fixture
def settings(tmp_path: Path, corpus: Path) -> Settings:
    return Settings(
        _env_file=None, corpus_dir=corpus, data_dir=tmp_path / "data", chunk_size=12, chunk_overlap=2
    )


def test_subjects() -> None:
    assert subject_for("Obstentrics_Williams") == "obstetrics"
    assert subject_for("Unknown_Book") == "general"


def test_load_books_filter_and_missing(corpus: Path) -> None:
    assert [b.name for b in load_books(corpus, ["anatomy_gray"])] == ["Anatomy_Gray"]
    with pytest.raises(FileNotFoundError):
        load_books(corpus, ["Nope_Book"])


def test_ingest_then_answer(settings: Settings, corpus: Path) -> None:
    store = QdrantStore(QdrantClient(":memory:"), "test")
    embedder = KeywordEmbedder()
    stats = ingest(
        load_books(corpus),
        settings=settings,
        offsets_fn=whitespace_offsets,
        embedder=embedder,
        store=store,
    )
    assert stats.books == 2 and stats.chunks == store.count() > 2

    # chunks.jsonl mirrors the index and slices match the source text
    lines = (settings.data_dir / "chunks.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    source = (corpus / f"{first['book']}.txt").read_text()
    assert source[first["start"] : first["end"]] == first["text"]

    llm = EchoLLM()
    rag = RAGPipeline(embedder, store, llm, top_k=2)
    ans = rag.answer("What treats absence seizure?")
    assert {h.chunk.book for h in ans.hits} == {"Pharmacology_Katzung"}
    user_prompt = llm.prompts[0][1].content
    assert user_prompt.startswith("Passages:") and "[1] (Source: Pharmacology_Katzung)" in user_prompt

    ok = grade(SmokeCase("t", "q", keywords=("ethosuximide",), books=("Pharmacology_Katzung",)), ans)
    assert ok.answer_ok and ok.retrieval_ok


def test_reingest_replaces_instead_of_duplicating(settings: Settings, corpus: Path) -> None:
    store = QdrantStore(QdrantClient(":memory:"), "test")
    for _ in range(2):
        stats = ingest(
            load_books(corpus),
            settings=settings,
            offsets_fn=whitespace_offsets,
            embedder=KeywordEmbedder(),
            store=store,
        )
    assert store.count() == stats.chunks


def test_grade_abstention(settings: Settings) -> None:
    rag = RAGPipeline(KeywordEmbedder(), QdrantStore(QdrantClient(":memory:"), "t"), EchoLLM(NOT_FOUND), 1)
    rag.store.recreate(KeywordEmbedder.dim)
    ans = rag.answer("capital of Australia?")
    assert grade(SmokeCase("oos", "q", answerable=False), ans).answer_ok
    assert not grade(SmokeCase("x", "q", keywords=("canberra",)), ans).answer_ok


def test_abstention_paraphrases() -> None:
    assert abstained("I could not find the information about that in the provided passages.")
    assert abstained("The passages do not contain this information.")
    assert not abstained("Ethosuximide is first line [1].")
