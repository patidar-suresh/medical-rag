"""LLM client abstraction.

The pipeline depends only on the `LLMClient` protocol, so the serving backend
(Ollama today, vLLM or llama.cpp server later) can be swapped by configuration.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import ollama

from medrag.config import Settings

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    # Why generation stopped, e.g. "stop" or "length" (truncated by num_predict/num_ctx).
    done_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    # Server-side timings: time to first token (model load + prompt processing) and decode speed.
    ttft_s: float | None = None
    tokens_per_s: float | None = None

    @property
    def truncated(self) -> bool:
        return self.done_reason == "length"


class LLMClient(Protocol):
    model: str

    def generate(self, messages: list[Message], *, json_schema: dict[str, Any] | None = None) -> Completion:
        """Return the full completion. With `json_schema`, the output is constrained to it."""
        ...

    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield text fragments as they are generated."""
        ...


class OllamaClient:
    """`LLMClient` backed by a local Ollama server."""

    def __init__(self, settings: Settings, client: ollama.Client | None = None) -> None:
        self.model = settings.llm_model
        self._settings = settings
        self._client = client or ollama.Client(host=settings.ollama_url, timeout=settings.llm_timeout_s)

    def _options(self) -> dict[str, Any]:
        # Set num_ctx explicitly: Ollama's default context is small and it
        # silently truncates long prompts instead of raising an error.
        return {"num_ctx": self._settings.llm_num_ctx, "temperature": self._settings.llm_temperature}

    @staticmethod
    def _to_wire(messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def generate(self, messages: list[Message], *, json_schema: dict[str, Any] | None = None) -> Completion:
        resp = self._client.chat(
            model=self.model,
            messages=self._to_wire(messages),
            format=json_schema,
            options=self._options(),
            keep_alive=self._settings.llm_keep_alive,
            stream=False,
        )
        ns = 1e9
        ttft = None
        if resp.prompt_eval_duration is not None:
            ttft = ((resp.load_duration or 0) + resp.prompt_eval_duration) / ns
        tps = None
        if resp.eval_count and resp.eval_duration:
            tps = resp.eval_count / (resp.eval_duration / ns)
        return Completion(
            text=resp.message.content or "",
            model=resp.model or self.model,
            done_reason=resp.done_reason,
            prompt_tokens=resp.prompt_eval_count,
            completion_tokens=resp.eval_count,
            ttft_s=ttft,
            tokens_per_s=tps,
        )

    def stream(self, messages: list[Message]) -> Iterator[str]:
        for chunk in self._client.chat(
            model=self.model,
            messages=self._to_wire(messages),
            options=self._options(),
            keep_alive=self._settings.llm_keep_alive,
            stream=True,
        ):
            if chunk.message.content:
                yield chunk.message.content

    def unload(self) -> None:
        """Free the model's GPU memory now (e.g. before bulk embedding on a small GPU)."""
        self._client.generate(model=self.model, prompt="", keep_alive=0)

    def installed_models(self) -> list[str]:
        return [m.model for m in self._client.list().models if m.model]
