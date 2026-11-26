"""Typed application settings, loaded from environment variables and `.env`.

Every setting can be overridden with an env var prefixed `MEDRAG_`, e.g.
`MEDRAG_LLM_MODEL=qwen3:8b`. No cloud API keys exist here by design.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEDRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Local LLM (Ollama)
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    llm_num_ctx: int = Field(default=8192, ge=512)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_keep_alive: str = "10m"
    llm_timeout_s: float = Field(default=120.0, gt=0)

    # Vector database (Qdrant)
    qdrant_url: str = "http://localhost:6333"
    collection: str = "medrag_chunks"

    # Embeddings (used from Phase 1)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_device: str = "cuda"

    # Data
    corpus_dir: Path = Path("textbooks/en")
    data_dir: Path = Path("data")

    # Logging
    log_level: str = "INFO"
    log_json: bool = False


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton. Tests construct `Settings(...)` directly."""
    return Settings()
