from pathlib import Path

import pytest

from medrag.config import Settings


def test_defaults_are_local(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("MEDRAG_OLLAMA_URL", "MEDRAG_QDRANT_URL", "MEDRAG_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.ollama_url.startswith("http://localhost")
    assert s.qdrant_url.startswith("http://localhost")
    assert not any("api_key" in name for name in Settings.model_fields)


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDRAG_LLM_MODEL", "qwen3:8b")
    monkeypatch.setenv("MEDRAG_LLM_NUM_CTX", "4096")
    s = Settings(_env_file=None)
    assert s.llm_model == "qwen3:8b"
    assert s.llm_num_ctx == 4096


def test_invalid_temperature_rejected() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, llm_temperature=5.0)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_corpus_dir_is_path() -> None:
    assert isinstance(Settings(_env_file=None).corpus_dir, Path)
