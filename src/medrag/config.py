"""Typed application settings, loaded from environment variables and `.env`.

Every setting can be overridden with an env var prefixed `MEDRAG_`, e.g.
`MEDRAG_LLM_MODEL=qwen3:8b`. No cloud API keys exist here by design.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
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

    # Embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_device: str = "cuda"  # falls back to CPU when CUDA is unavailable
    embed_batch_size: int = Field(default=64, ge=1)
    # bge-*-v1.5 models expect this instruction on queries (not on passages).
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Chunking (Phase 1: fixed-size token windows). Must stay below the
    # embedding model's max sequence length (512 for bge-small, incl. 2 special tokens).
    chunk_size: int = Field(default=500, ge=8, le=510)
    chunk_overlap: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})")
        return self

    # Retrieval
    top_k: int = Field(default=5, ge=1, le=50)

    # Evaluation: open NLI model used to check answer sentences against passages
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    # Entailment probability above which a sentence counts as supported.
    # Calibrate against human labels with `medrag eval label` + `medrag eval calibrate`.
    nli_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)

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
