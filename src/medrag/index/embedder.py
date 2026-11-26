"""Local sentence-embedding model (sentence-transformers)."""

from collections.abc import Sequence
from typing import Protocol

from medrag.config import Settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str], *, show_progress: bool = False) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, settings: Settings) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        from medrag.hf import load_cached_first, quiet_hf

        quiet_hf()
        device = settings.embed_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self._model = load_cached_first(
            lambda local: SentenceTransformer(settings.embed_model, device=device, local_files_only=local)
        )
        if device.startswith("cuda"):
            self._model.half()  # fp16: half the VRAM, faster, negligible quality loss for retrieval
        self._batch_size = settings.embed_batch_size
        self._query_prefix = settings.embed_query_prefix
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise ValueError(f"Cannot determine embedding size of {settings.embed_model}")
        self.dim: int = dim

    def embed_documents(self, texts: Sequence[str], *, show_progress: bool = False) -> list[list[float]]:
        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,  # cosine similarity == dot product
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return [[float(x) for x in v] for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self._query_prefix + text])[0]
