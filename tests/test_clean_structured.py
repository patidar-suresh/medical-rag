"""Cleaning rules, structured chunking and parent-child retrieval."""

from pathlib import Path

from fakes import EchoLLM, KeywordEmbedder, whitespace_offsets
from qdrant_client import QdrantClient

from medrag.config import Settings
from medrag.eval.common import compact
from medrag.index.store import QdrantStore
from medrag.ingest.clean import (
    Block,
    build_vocab,
    clean_text,
    dehyphenate,
    normalize_unicode,
    parse_book,
    split_long,
)
from medrag.ingest.loader import load_books
from medrag.ingest.pipeline import chunk_book, ingest
from medrag.ingest.structured import make_units, pack
from medrag.rag import RAGPipeline
from medrag.retrieval import RetrievalOptions

PARA = "The deltoid muscle is supplied by the axillary nerve. " * 12  # ~ a real paragraph (>150 chars)


def test_unicode_ligatures_spaces_and_layout_residue() -> None:
    raw = "ﬁbrosis of the­liver Brunicardi_Ch03_p0083-p0102.indd 10108/12/18 10:07 AM next"
    assert normalize_unicode(raw) == "fibrosis of theliver next"


def test_dehyphenate_uses_book_vocabulary() -> None:
    vocab = build_vocab("statistical manual. hair-pulling disorder.")
    assert dehyphenate("Statisti- cal and hair- pulling", vocab) == "Statistical and hair-pulling"
    assert dehyphenate("pre- and postoperative", vocab) == "pre- and postoperative"


def test_parse_book_rules() -> None:
    raw = "\n".join(
        [
            "Chapter 11 Clinical Problems of Aging",  # running header, repeated below
            "Neoplasia Basics",  # heading: short, no period, followed by a paragraph
            PARA,
            "Fig. 1.1",  # figure label only -> dropped
            "Fig. 1.2 The organization of nuclear DNA in the cell nucleus.",  # caption -> kept
            "2.",  # lone list marker -> merged into next line
            "",
            "What is the composition of the plasma membrane?",
            "The receptor binds its ligand and",  # broken paragraph...
            "then signals downstream.",  # ...continued in lowercase -> joined
            "Chapter 11 Clinical Problems of Aging",
            "Chapter 11 Clinical Problems of Aging",
            "12",  # page number -> dropped
        ]
    )
    blocks = parse_book(raw)
    texts = [b.text for b in blocks]
    assert texts.count("Chapter 11 Clinical Problems of Aging") == 1
    assert Block("heading", "Neoplasia Basics", 2) in blocks
    assert "Fig. 1.1" not in texts and any(t.startswith("Fig. 1.2 The organization") for t in texts)
    assert "2. What is the composition of the plasma membrane?" in texts
    assert "The receptor binds its ligand and then signals downstream." in texts
    assert "12" not in texts


def test_split_long_paragraphs() -> None:
    sentences = "Sentence number one is here. " * 200
    assert all(len(p) <= 3000 for p in split_long(sentences))
    no_breaks = "word " * 2000  # a table-like span with no sentence ends
    assert all(len(p) <= 3000 for p in split_long(no_breaks))


def test_pack_respects_budget_overlap_and_headings() -> None:
    blocks = [Block("heading", "Section A", 2)] + [
        Block("paragraph", f"alpha beta gamma delta {i}.") for i in range(6)
    ]
    blocks += [Block("heading", "Section B", 2), Block("paragraph", "epsilon zeta eta theta iota.")]
    units = make_units(blocks, whitespace_offsets, max_tokens=12)
    ranges = pack(units, 0, len(units), budget=12, overlap=5)
    for a, b in ranges:
        assert sum(u.n_tokens for u in units[a:b]) <= 12 or b - a == 1
    starts_b = [a for a, _ in ranges if units[a].is_heading and "B" in str(units[a].crumb)]
    assert starts_b, "a new chunk starts at heading Section B"
    assert any(ranges[i][0] < ranges[i - 1][1] for i in range(1, len(ranges))), "some overlap carried"


def test_structured_chunks_slice_cleaned_text_and_carry_breadcrumb(tmp_path: Path) -> None:
    (tmp_path / "Anatomy_Gray.txt").write_text("Upper Limb\n" + PARA + "\n" + PARA)
    [book] = load_books(tmp_path)
    s = Settings(_env_file=None, chunk_strategy="structured", chunk_size=60, chunk_overlap=0)
    chunks = chunk_book(book, whitespace_offsets, s)
    text = clean_text(parse_book(book.text))
    assert len(chunks) > 1
    assert all(text[c.start : c.end] == c.text for c in chunks)
    assert all(c.context.startswith("Gray's Anatomy") for c in chunks)
    assert chunks[0].context == "Gray's Anatomy › Upper Limb"
    assert chunks[0].embed_text(True).startswith("Gray's Anatomy › Upper Limb\n\n")


def test_parent_child_ingest_and_retrieval(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "Anatomy_Gray.txt").write_text("Upper Limb\n" + PARA * 3)
    (corpus / "Pharmacology_Katzung.txt").write_text(
        "Antiseizure Drugs\n" + "Ethosuximide is first line for absence seizure in children. " * 30
    )
    s = Settings(
        _env_file=None,
        corpus_dir=corpus,
        data_dir=tmp_path / "data",
        chunk_strategy="structured",
        contextual_headers=False,
        chunk_size=40,
        chunk_overlap=0,
        parent_chunk_size=200,
    )
    store = QdrantStore(QdrantClient(":memory:"), "pc")
    emb = KeywordEmbedder()
    ingest(load_books(corpus), settings=s, offsets_fn=whitespace_offsets, embedder=emb, store=store)
    rag = RAGPipeline(emb, store, EchoLLM(), top_k=2, options=RetrievalOptions(parent_mode=True))
    hits = rag.retrieve("absence seizure ethosuximide")
    assert len({h.chunk.chunk_id for h in hits}) == len(hits)  # parents deduplicated
    assert hits[0].chunk.book == "Pharmacology_Katzung"
    assert "#" in hits[0].chunk.chunk_id and "." not in hits[0].chunk.chunk_id.split("#")[1]  # a parent id
    assert hits[0].chunk.n_tokens > 40  # parent text, not the 40-token child


def test_compact_matches_across_cleaning() -> None:
    assert compact("Statisti- cal Manual of ﬁbrosis") in compact("the Statistical manual of fibrosis.")
