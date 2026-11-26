from pathlib import Path

from test_llm import make

from medrag.config import Settings
from medrag.doctor import check_corpus, check_ollama, check_qdrant


def test_ollama_ok_when_model_installed() -> None:
    llm, _ = make()
    r = check_ollama(Settings(_env_file=None, llm_model="qwen3:8b"), llm)
    assert r.ok


def test_ollama_fails_with_pull_hint_when_model_missing() -> None:
    llm, _ = make()
    r = check_ollama(Settings(_env_file=None, llm_model="gemma3:4b"), llm)
    assert not r.ok
    assert "ollama pull gemma3:4b" in r.detail


def test_qdrant_unreachable_is_reported_not_raised() -> None:
    r = check_qdrant(Settings(_env_file=None, qdrant_url="http://127.0.0.1:1"))
    assert not r.ok
    assert "docker compose up" in r.detail


def test_corpus(tmp_path: Path) -> None:
    assert not check_corpus(Settings(_env_file=None, corpus_dir=tmp_path)).ok
    (tmp_path / "Book.txt").write_text("text")
    r = check_corpus(Settings(_env_file=None, corpus_dir=tmp_path))
    assert r.ok and "1 books" in r.detail
