"""Local sentence-embedding model (sentence-transformers)."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from medrag.config import Settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str], *, show_progress: bool = False) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _load(name: str, pooling: str, max_length: int, device: str) -> "SentenceTransformer":
    """A sentence-transformers model, or a plain HF encoder wrapped with the given pooling."""
    from sentence_transformers import SentenceTransformer

    from medrag.hf import load_cached_first

    def load(local: bool) -> Any:
        if pooling == "auto":
            m = SentenceTransformer(name, device=device, local_files_only=local)
            m.max_seq_length = min(m.max_seq_length or max_length, max_length)
            return m
        from sentence_transformers.sentence_transformer.modules import Pooling, Transformer

        kw = {"local_files_only": local}
        enc = Transformer(
            name, model_kwargs=kw, processor_kwargs=kw, config_kwargs=kw, max_seq_length=max_length
        )
        return SentenceTransformer(
            modules=[enc, Pooling(enc.get_embedding_dimension(), pooling_mode=pooling)], device=device
        )

    model: SentenceTransformer = load_cached_first(load)
    if device.startswith("cuda"):
        model.half()  # fp16: half the VRAM, faster, negligible quality loss for retrieval
    return model


class SentenceTransformerEmbedder:
    def __init__(self, settings: Settings) -> None:
        import torch

        from medrag.hf import quiet_hf

        quiet_hf()
        device = settings.embed_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        args = (settings.embed_pooling, settings.embed_max_length, device)
        self._model = _load(settings.embed_model, *args)
        # Asymmetric models (MedCPT) have a separate, query-tuned encoder.
        self._query_model = _load(settings.embed_query_model, *args) if settings.embed_query_model else None
        self._batch_size = settings.embed_batch_size
        self._query_prefix = settings.embed_query_prefix
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise ValueError(f"Cannot determine embedding size of {settings.embed_model}")
        self.dim: int = dim

    def _encode(
        self, model: "SentenceTransformer", texts: Sequence[str], show_progress: bool
    ) -> list[list[float]]:
        vecs = model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,  # cosine similarity == dot product
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return [[float(x) for x in v] for v in vecs]

    def embed_documents(self, texts: Sequence[str], *, show_progress: bool = False) -> list[list[float]]:
        return self._encode(self._model, texts, show_progress)

    def embed_query(self, text: str) -> list[float]:
        return self._encode(self._query_model or self._model, [self._query_prefix + text], False)[0]
