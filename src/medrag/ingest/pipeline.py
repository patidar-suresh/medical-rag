"""Offline ingestion: books -> chunks -> embeddings -> Qdrant (+ chunks.jsonl)."""

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from medrag.config import Settings
from medrag.index.embedder import Embedder
from medrag.index.store import Chunk, QdrantStore
from medrag.ingest.chunker import OffsetsFn, chunk_spans, token_offsets
from medrag.ingest.loader import Book
from medrag.logs import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class IngestStats:
    books: int
    chunks: int
    tokens: int
    seconds_chunking: float
    seconds_embedding: float
    seconds_indexing: float


def chunk_books(books: Sequence[Book], offsets_fn: OffsetsFn, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for book in books:
        spans = chunk_spans(token_offsets(book.text, offsets_fn), size, overlap)
        chunks.extend(
            Chunk(
                chunk_id=f"{book.name}#{i}",
                book=book.name,
                subject=book.subject,
                text=book.text[s.start : s.end],
                start=s.start,
                end=s.end,
                n_tokens=s.n_tokens,
            )
            for i, s in enumerate(spans)
        )
        log.info("chunked", book=book.name, chunks=len(spans))
    return chunks


def write_chunks(chunks: Sequence[Chunk], path: Path) -> None:
    """Keep a plain copy of every chunk; the eval harness (Phase 2) samples from it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def ingest(
    books: Sequence[Book],
    *,
    settings: Settings,
    offsets_fn: OffsetsFn,
    embedder: Embedder,
    store: QdrantStore,
    show_progress: bool = False,
) -> IngestStats:
    t0 = time.perf_counter()
    chunks = chunk_books(books, offsets_fn, settings.chunk_size, settings.chunk_overlap)
    write_chunks(chunks, settings.data_dir / "chunks.jsonl")
    t1 = time.perf_counter()

    vectors = embedder.embed_documents([c.text for c in chunks], show_progress=show_progress)
    t2 = time.perf_counter()

    # Phase 1 rebuilds the collection from scratch; incremental re-indexing comes in Phase 8.
    store.recreate(embedder.dim)
    store.upsert(chunks, vectors)
    t3 = time.perf_counter()

    return IngestStats(
        books=len(books),
        chunks=len(chunks),
        tokens=sum(c.n_tokens for c in chunks),
        seconds_chunking=t1 - t0,
        seconds_embedding=t2 - t1,
        seconds_indexing=t3 - t2,
    )
