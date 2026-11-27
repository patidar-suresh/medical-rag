"""Cross-encoder reranker: reads (query, passage) together, so it scores relevance far more
precisely than comparing two independently computed embeddings. Too slow for the whole corpus,
so it only reorders the top candidates from the first-stage search."""

from collections.abc import Sequence
from typing import Protocol

from medrag.config import Settings


class Reranker(Protocol):
    def scores(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, settings: Settings) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        from medrag.hf import load_cached_first, quiet_hf

        quiet_hf()
        device = settings.embed_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.model_name = settings.rerank_model
        self._model = load_cached_first(
            lambda local: CrossEncoder(
                settings.rerank_model,
                device=device,
                max_length=settings.rerank_max_length,
                local_files_only=local,
            )
        )
        if device.startswith("cuda"):
            self._model.model.half()
        self._batch_size = settings.rerank_batch_size
        # Rank by raw logits rather than sigmoid probabilities, which saturate near 0 and 1
        # (same order, but readable scores).
        self._activation = torch.nn.Identity()

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        out = self._model.predict(
            [(query, p) for p in passages],
            batch_size=self._batch_size,
            activation_fn=self._activation,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(x) for x in out]
