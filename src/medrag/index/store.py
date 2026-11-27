"""Qdrant store for textbook chunks: a dense vector and (optionally) a BM25 sparse vector each."""

import uuid
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass

from qdrant_client import QdrantClient, models

from medrag.index.sparse import SparseVector

DENSE = "dense"
SPARSE = "bm25"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<book>#<n>"
    book: str
    subject: str
    text: str
    start: int  # char offsets into the book text (cleaned text when cleaning is on)
    end: int
    n_tokens: int
    context: str = ""  # breadcrumb "Book › Chapter › Section" (structured chunking)
    content_hash: str = ""  # sha1 of text, for change detection on re-ingest
    # Parent-child ("small-to-big"): this chunk is embedded, its parent is what gets returned.
    parent_id: str | None = None
    parent_text: str | None = None
    parent_n_tokens: int | None = None

    def embed_text(self, contextual: bool) -> str:
        """What gets embedded: the text, optionally prefixed by its breadcrumb."""
        return f"{self.context}\n\n{self.text}" if contextual and self.context else self.text

    def as_parent(self) -> "Chunk":
        """The chunk to show the LLM: the parent section when there is one."""
        if self.parent_id is None or self.parent_text is None:
            return self
        return Chunk(
            chunk_id=self.parent_id,
            book=self.book,
            subject=self.subject,
            text=self.parent_text,
            start=self.start,
            end=self.end,
            n_tokens=self.parent_n_tokens or self.n_tokens,
            context=self.context,
        )


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def point_id(chunk_id: str) -> str:
    """Deterministic UUID, so re-ingesting a chunk overwrites it instead of duplicating it."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"medrag:{chunk_id}"))


def _batches(n: int, size: int) -> Iterator[slice]:
    for i in range(0, n, size):
        yield slice(i, i + size)


def _sparse(v: SparseVector) -> models.SparseVector:
    return models.SparseVector(indices=v.indices, values=v.values)


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self.collection = collection
        self._layout: tuple[str | None, bool] | None = None  # (dense vector name, has sparse)

    @classmethod
    def from_url(cls, url: str, collection: str) -> "QdrantStore":
        return cls(QdrantClient(url=url, timeout=60, check_compatibility=False), collection)

    def recreate(self, dim: int, sparse: bool = True) -> None:
        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)
        self._client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            # Qdrant computes IDF from the collection; we store BM25 term-frequency weights.
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)}
            if sparse
            else None,
            on_disk_payload=True,  # chunk text lives in the payload; keep RAM for vectors
        )
        self._layout = (DENSE, sparse)

    def layout(self) -> tuple[str | None, bool]:
        """(dense vector name, has sparse). Pre-Phase-4 collections have one unnamed dense vector."""
        if self._layout is None:
            params = self._client.get_collection(self.collection).config.params
            dense = DENSE if isinstance(params.vectors, dict) else None
            self._layout = (dense, bool(params.sparse_vectors and SPARSE in params.sparse_vectors))
        return self._layout

    @property
    def has_sparse(self) -> bool:
        return self.layout()[1]

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
        sparse: Sequence[SparseVector] | None = None,
        batch_size: int = 256,
    ) -> None:
        if len(chunks) != len(vectors) or (sparse is not None and len(sparse) != len(chunks)):
            raise ValueError("chunks and vectors must have the same length")
        dense_name, _ = self.layout()

        def vector(i: int) -> models.VectorStruct:
            if dense_name is None:
                return vectors[i]
            v: dict[str, list[float] | models.SparseVector] = {dense_name: vectors[i]}
            if sparse is not None:
                v[SPARSE] = _sparse(sparse[i])
            return v  # type: ignore[return-value]

        for s in _batches(len(chunks), batch_size):
            self._client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=point_id(chunks[i].chunk_id), vector=vector(i), payload=asdict(chunks[i])
                    )
                    for i in range(*s.indices(len(chunks)))
                ],
                wait=True,
            )

    @staticmethod
    def _hits(points: Sequence[models.ScoredPoint]) -> list[Hit]:
        return [Hit(Chunk(**(p.payload or {})), p.score) for p in points]

    def search(self, vector: list[float], k: int) -> list[Hit]:
        res = self._client.query_points(
            self.collection, query=vector, using=self.layout()[0], limit=k, with_payload=True
        )
        return self._hits(res.points)

    def search_sparse(self, vector: SparseVector, k: int) -> list[Hit]:
        if not self.has_sparse:
            raise ValueError(f"Collection '{self.collection}' has no BM25 vectors; re-ingest it.")
        res = self._client.query_points(
            self.collection, query=_sparse(vector), using=SPARSE, limit=k, with_payload=True
        )
        return self._hits(res.points)

    def search_both(self, dense: list[float], sparse: SparseVector, k: int) -> tuple[list[Hit], list[Hit]]:
        """Dense and BM25 results in one round trip, for client-side fusion."""
        if not self.has_sparse:
            raise ValueError(f"Collection '{self.collection}' has no BM25 vectors; re-ingest it.")
        a, b = self._client.query_batch_points(
            self.collection,
            requests=[
                models.QueryRequest(query=dense, using=self.layout()[0], limit=k, with_payload=True),
                models.QueryRequest(query=_sparse(sparse), using=SPARSE, limit=k, with_payload=True),
            ],
        )
        return self._hits(a.points), self._hits(b.points)

    def exists(self) -> bool:
        return self._client.collection_exists(self.collection)

    def count(self) -> int:
        return self._client.count(self.collection, exact=True).count
