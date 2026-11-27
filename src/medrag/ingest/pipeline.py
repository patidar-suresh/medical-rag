"""Offline ingestion: books -> (clean) -> chunks -> embeddings -> Qdrant (+ chunks jsonl)."""

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from medrag.config import Settings
from medrag.index.embedder import Embedder
from medrag.index.sparse import BM25Encoder
from medrag.index.store import Chunk, QdrantStore
from medrag.ingest.chunker import OffsetsFn, chunk_spans, token_offsets
from medrag.ingest.clean import clean_text, parse_book
from medrag.ingest.loader import Book
from medrag.ingest.structured import make_units, pack, to_pieces
from medrag.logs import get_logger

log = get_logger(__name__)

# Token room kept for the contextual header inside the embedding model's 512-token window.
HEADER_RESERVE = 48
MAX_CONTEXT_CHARS = 160


@dataclass(frozen=True)
class IngestStats:
    books: int
    chunks: int
    tokens: int
    seconds_chunking: float
    seconds_embedding: float
    seconds_indexing: float


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _context(text: str) -> str:
    return text if len(text) <= MAX_CONTEXT_CHARS else text[: MAX_CONTEXT_CHARS - 1] + "…"


def chunk_book(book: Book, offsets_fn: OffsetsFn, s: Settings) -> list[Chunk]:
    # Structured chunking needs the parsed blocks, so it always works on cleaned text.
    blocks = parse_book(book.text) if s.clean_text or s.chunk_strategy == "structured" else None
    text = clean_text(blocks) if blocks is not None else book.text
    reserve = HEADER_RESERVE if s.contextual_headers else 0

    def make(
        i: str,
        start: int,
        end: int,
        n: int,
        ctx: str,
        parent_id: str | None = None,
        parent_text: str | None = None,
        parent_n_tokens: int | None = None,
    ) -> Chunk:
        body = text[start:end]
        return Chunk(
            chunk_id=f"{book.name}#{i}",
            book=book.name,
            subject=book.subject,
            text=body,
            start=start,
            end=end,
            n_tokens=n,
            context=_context(ctx),
            content_hash=_hash(body),
            parent_id=parent_id,
            parent_text=parent_text,
            parent_n_tokens=parent_n_tokens,
        )

    if s.chunk_strategy == "fixed":
        spans = chunk_spans(token_offsets(text, offsets_fn), s.chunk_size - reserve, s.chunk_overlap)
        return [make(str(i), sp.start, sp.end, sp.n_tokens, book.title) for i, sp in enumerate(spans)]

    assert blocks is not None
    budget = s.chunk_size - reserve
    units = make_units(blocks, offsets_fn, budget)
    if not s.parent_chunk_size:
        pieces = to_pieces(units, pack(units, 0, len(units), budget, s.chunk_overlap), book.title)
        return [make(str(i), p.start, p.end, p.n_tokens, p.context) for i, p in enumerate(pieces)]

    # Parent-child: parents first, then children packed inside each parent.
    chunks: list[Chunk] = []
    parents = to_pieces(units, pack(units, 0, len(units), s.parent_chunk_size, 0), book.title)
    for pi, p in enumerate(parents):
        parent_text = text[p.start : p.end]
        kids = to_pieces(units, pack(units, p.units[0], p.units[1], budget, s.chunk_overlap), book.title)
        for ci, k in enumerate(kids):
            chunks.append(
                make(
                    f"{pi}.{ci}",
                    k.start,
                    k.end,
                    k.n_tokens,
                    k.context,
                    parent_id=f"{book.name}#{pi}",
                    parent_text=parent_text,
                    parent_n_tokens=p.n_tokens,
                )
            )
    return chunks


def chunk_books(books: Sequence[Book], offsets_fn: OffsetsFn, settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for book in books:
        book_chunks = chunk_book(book, offsets_fn, settings)
        chunks.extend(book_chunks)
        log.info("chunked", book=book.name, chunks=len(book_chunks))
    return chunks


def write_chunks(chunks: Sequence[Chunk], path: Path) -> None:
    """Keep a plain copy of every chunk; the eval harness samples from it."""
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
    chunks = chunk_books(books, offsets_fn, settings)
    write_chunks(chunks, settings.chunks_path)
    t1 = time.perf_counter()

    texts = [c.embed_text(settings.contextual_headers) for c in chunks]
    vectors = embedder.embed_documents(texts, show_progress=show_progress)
    sparse = BM25Encoder().fit(texts).encode_documents(texts) if settings.index_sparse else None
    t2 = time.perf_counter()

    # Rebuilds the collection from scratch; incremental re-indexing comes in Phase 8.
    store.recreate(embedder.dim, sparse=settings.index_sparse)
    store.upsert(chunks, vectors, sparse)
    t3 = time.perf_counter()

    return IngestStats(
        books=len(books),
        chunks=len(chunks),
        tokens=sum(c.n_tokens for c in chunks),
        seconds_chunking=t1 - t0,
        seconds_embedding=t2 - t1,
        seconds_indexing=t3 - t2,
    )
