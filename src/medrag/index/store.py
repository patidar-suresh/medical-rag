"""Qdrant vector store for textbook chunks."""

import uuid
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass

from qdrant_client import QdrantClient, models


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<book>#<n>"
    book: str
    subject: str
    text: str
    start: int  # char offsets into the book file
    end: int
    n_tokens: int


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


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self.collection = collection

    @classmethod
    def from_url(cls, url: str, collection: str) -> "QdrantStore":
        return cls(QdrantClient(url=url, timeout=60, check_compatibility=False), collection)

    def recreate(self, dim: int) -> None:
        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)
        self._client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            on_disk_payload=True,  # chunk text lives in the payload; keep RAM for vectors
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]], batch_size: int = 256) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for s in _batches(len(chunks), batch_size):
            self._client.upsert(
                self.collection,
                points=[
                    models.PointStruct(id=point_id(c.chunk_id), vector=v, payload=asdict(c))
                    for c, v in zip(chunks[s], vectors[s], strict=True)
                ],
                wait=True,
            )

    def search(self, vector: list[float], k: int) -> list[Hit]:
        res = self._client.query_points(self.collection, query=vector, limit=k, with_payload=True)
        return [Hit(Chunk(**(p.payload or {})), p.score) for p in res.points]

    def exists(self) -> bool:
        return self._client.collection_exists(self.collection)

    def count(self) -> int:
        return self._client.count(self.collection, exact=True).count
