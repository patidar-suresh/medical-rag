"""Lightweight fakes so pipeline tests run without models, GPUs or servers."""

import re
from collections.abc import Iterator, Sequence
from typing import Any

from medrag.generate.llm import Completion, Message


def whitespace_offsets(texts: Sequence[str]) -> list[list[tuple[int, int]]]:
    """Stand-in for a real tokenizer: every whitespace-separated word is one token."""
    return [[(m.start(), m.end()) for m in re.finditer(r"\S+", t)] for t in texts]


class KeywordEmbedder:
    """Deterministic bag-of-keywords vectors: texts sharing vocabulary end up close."""

    VOCAB = ["seizure", "ethosuximide", "heart", "lung", "nerve", "deltoid", "axillary", "liver"]
    dim = len(VOCAB)

    def _vec(self, text: str) -> list[float]:
        t = text.lower()
        v = [float(t.count(w)) for w in self.VOCAB]
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v] if any(v) else [1.0 / self.dim**0.5] * self.dim

    def embed_documents(self, texts: Sequence[str], *, show_progress: bool = False) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class EchoLLM:
    """Records prompts and returns a fixed answer."""

    model = "fake"

    def __init__(self, reply: str = "Ethosuximide [1].") -> None:
        self.reply = reply
        self.prompts: list[list[Message]] = []

    def generate(self, messages: list[Message], *, json_schema: dict[str, Any] | None = None) -> Completion:
        self.prompts.append(messages)
        return Completion(self.reply, self.model, "stop", 10, 5)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        self.prompts.append(messages)
        yield from self.reply.split(" ")
