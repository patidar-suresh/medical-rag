"""Phase 4 retrieval: BM25 vectors, RRF fusion, reranking, diversity cap, abbreviations."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from fakes import EchoLLM, KeywordEmbedder, whitespace_offsets
from qdrant_client import QdrantClient, models

from medrag.config import Settings
from medrag.index.sparse import BM25Encoder
from medrag.index.store import Chunk, Hit, QdrantStore
from medrag.ingest.loader import load_books
from medrag.ingest.pipeline import ingest
from medrag.query import expand_abbreviations
from medrag.rag import RAGPipeline
from medrag.retrieval import RetrievalOptions, Retriever, cap_per_book, rrf


def hit(cid: str, book: str = "B", score: float = 0.0) -> Hit:
    return Hit(Chunk(cid, book, "s", cid, 0, 1, 1), score)


# --------------------------------------------------------------------------- BM25


def test_bm25_tokens_stem_and_drop_stopwords() -> None:
    enc = BM25Encoder()
    assert enc.tokens("The seizures of a beta-blocker") == ["seizur", "beta", "blocker"]
    assert enc.tokens("Seizure") == enc.tokens("seizures")


def test_bm25_length_normalisation_and_saturation() -> None:
    enc = BM25Encoder(avgdl=10)
    short = enc.encode_document("ethosuximide seizure")
    long = enc.encode_document("ethosuximide " + " ".join(f"w{i}" for i in range(40)))
    term = enc.encode_query("ethosuximide").indices[0]

    def weight(v: object) -> float:
        return dict(zip(v.indices, v.values, strict=True))[term]  # type: ignore[attr-defined]

    assert weight(short) > weight(long)  # same tf, longer doc -> lower weight
    many = enc.encode_document("ethosuximide " * 50)
    assert weight(many) < enc.k1 + 1  # tf saturates at k1 + 1


def test_bm25_fit_sets_average_length() -> None:
    assert BM25Encoder().fit(["heart lung", "liver liver liver liver"]).avgdl == 3.0


# --------------------------------------------------------------------------- fusion & diversity


def test_rrf_prefers_agreement_between_lists() -> None:
    dense = [hit("a"), hit("b"), hit("c")]
    sparse = [hit("d"), hit("b"), hit("e")]
    fused = rrf([dense, sparse], k=60)
    assert fused[0].chunk.chunk_id == "b"  # #2 in both lists beats #1 in only one
    assert {h.chunk.chunk_id for h in fused} == {"a", "b", "c", "d", "e"}


def test_cap_per_book_is_soft() -> None:
    hits = [hit("1", "H"), hit("2", "H"), hit("3", "H"), hit("4", "G"), hit("5", "H")]
    assert [h.chunk.chunk_id for h in cap_per_book(hits, 2, 3)] == ["1", "2", "4"]
    # only one other book available: the gap is filled with the best skipped hits, in rank order
    assert [h.chunk.chunk_id for h in cap_per_book(hits, 1, 3)] == ["1", "2", "4"]
    assert cap_per_book(hits, 0, 2) == hits[:2]


# --------------------------------------------------------------------------- abbreviations


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("First-line drug after MI?", "First-line drug after MI (myocardial infarction)?"),
        ("Signs of DKA and HTN", "Signs of DKA (diabetic ketoacidosis) and HTN (hypertension)"),
        ("myocardial infarction (MI) risk", "myocardial infarction (MI) risk"),  # already spelled out
        ("MIS-C in children", "MIS-C in children"),  # whole words only
        ("Type IV hypersensitivity", "Type IV hypersensitivity"),  # Roman numerals untouched
        ("mi casa", "mi casa"),  # case-sensitive
    ],
)
def test_expand_abbreviations(query: str, expected: str) -> None:
    assert expand_abbreviations(query) == expected


# --------------------------------------------------------------------------- end to end


@pytest.fixture
def store(tmp_path: Path) -> QdrantStore:
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "Pharmacology_Katzung.txt").write_text(
        "Ethosuximide is first line for absence seizure. " * 3
        + "Digoxin toxicity causes yellow vision and arrhythmia. " * 3
    )
    (d / "Anatomy_Gray.txt").write_text("The deltoid is supplied by the axillary nerve. " * 6)
    s = Settings(
        _env_file=None,
        corpus_dir=d,
        data_dir=tmp_path / "data",
        chunk_strategy="fixed",
        clean_text=False,
        parent_chunk_size=0,
        chunk_size=8,
        chunk_overlap=0,
    )
    st = QdrantStore(QdrantClient(":memory:"), "t")
    ingest(load_books(d), settings=s, offsets_fn=whitespace_offsets, embedder=KeywordEmbedder(), store=st)
    return st


def test_sparse_finds_terms_the_embedder_cannot_see(store: QdrantStore) -> None:
    # "digoxin" is outside KeywordEmbedder's vocabulary: dense search is blind to it, BM25 is not.
    assert store.has_sparse
    r = Retriever(KeywordEmbedder(), store, RetrievalOptions(mode="sparse"))
    assert "Digoxin" in r.retrieve("digoxin toxicity", 1)[0].chunk.text
    hybrid = Retriever(KeywordEmbedder(), store, RetrievalOptions(mode="hybrid", candidates=10))
    top = hybrid.retrieve("digoxin toxicity", 3)
    assert any("Digoxin" in h.chunk.text for h in top)
    assert all(h.score <= 2 / 61 for h in top)  # RRF scores


class LengthReranker:
    """Fake cross-encoder: prefers passages mentioning 'axillary'."""

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float("axillary" in p) for p in passages]


def test_reranker_reorders_candidates(store: QdrantStore) -> None:
    rag = RAGPipeline(
        KeywordEmbedder(),
        store,
        EchoLLM(),
        top_k=2,
        options=RetrievalOptions(candidates=20),
        reranker=LengthReranker(),
    )
    hits = rag.retrieve("seizure")
    assert all("axillary" in h.chunk.text for h in hits) and hits[0].score == 1.0


def test_legacy_unnamed_collection_still_searchable() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        "old", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
    )
    st = QdrantStore(client, "old")
    st.upsert([Chunk("B#0", "B", "s", "x", 0, 1, 1)], [[1.0, 0.0]])
    assert st.layout() == (None, False)
    assert st.search([1.0, 0.0], 1)[0].chunk.chunk_id == "B#0"
    with pytest.raises(ValueError, match="no BM25"):
        Retriever(KeywordEmbedder(), st, RetrievalOptions(mode="hybrid")).retrieve("x", 1)
