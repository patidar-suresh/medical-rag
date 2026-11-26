"""Unit tests for OllamaClient using a fake ollama.Client (no server needed)."""

from collections.abc import Iterator
from typing import Any

from ollama import ChatResponse, ListResponse

from medrag.config import Settings
from medrag.generate.llm import Message, OllamaClient


class FakeOllama:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> ChatResponse | Iterator[ChatResponse]:
        self.calls.append(kwargs)
        if kwargs["stream"]:
            return iter(
                ChatResponse(model="m", message={"role": "assistant", "content": t})
                for t in ["Hel", "lo", ""]
            )
        return ChatResponse(
            model="m",
            message={"role": "assistant", "content": "Hello"},
            done_reason="length",
            prompt_eval_count=12,
            eval_count=3,
        )

    def list(self) -> ListResponse:
        return ListResponse(models=[{"model": "llama3.2:3b"}, {"model": "qwen3:8b"}])


def make(settings: Settings | None = None) -> tuple[OllamaClient, FakeOllama]:
    fake = FakeOllama()
    return OllamaClient(settings or Settings(_env_file=None), client=fake), fake  # type: ignore[arg-type]


def test_generate_maps_response_and_options() -> None:
    llm, fake = make(Settings(_env_file=None, llm_num_ctx=4096, llm_temperature=0.0))
    out = llm.generate([Message("system", "be brief"), Message("user", "hi")])
    assert out.text == "Hello"
    assert out.truncated
    assert (out.prompt_tokens, out.completion_tokens) == (12, 3)
    call = fake.calls[0]
    assert call["options"] == {"num_ctx": 4096, "temperature": 0.0}
    assert call["messages"][0] == {"role": "system", "content": "be brief"}
    assert call["format"] is None


def test_generate_passes_json_schema() -> None:
    llm, fake = make()
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    llm.generate([Message("user", "q")], json_schema=schema)
    assert fake.calls[0]["format"] == schema


def test_stream_skips_empty_fragments() -> None:
    llm, _ = make()
    assert list(llm.stream([Message("user", "hi")])) == ["Hel", "lo"]


def test_installed_models() -> None:
    llm, _ = make()
    assert llm.installed_models() == ["llama3.2:3b", "qwen3:8b"]
