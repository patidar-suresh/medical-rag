"""Typed application settings, loaded from environment variables and `.env`.

Every setting can be overridden with an env var prefixed `MEDRAG_`, e.g.
`MEDRAG_LLM_MODEL=qwen3:8b`. No cloud API keys exist here by design.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    collection: str = "medrag_pc160_800"  # one collection per chunking config (see chunks_path)

    # Embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_device: str = "cuda"  # falls back to CPU when CUDA is unavailable
    embed_batch_size: int = Field(default=64, ge=1)
    # bge-*-v1.5 models expect this instruction on queries (not on passages). Set it to ""
    # for models without one (bge-m3, MedCPT).
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    # Asymmetric models (MedCPT) encode queries with a separate model; "" = same as embed_model.
    embed_query_model: str = ""
    # "auto" = the model's own sentence-transformers config; "cls"/"mean" for plain HF encoders.
    embed_pooling: Literal["auto", "cls", "mean"] = "auto"
    embed_max_length: int = Field(default=512, ge=16, le=8192)
    # Tokenizer that measures chunk sizes; "" = embed_model's. Pin it to compare embedding
    # models on identical chunks.
    chunk_tokenizer: str = ""

    # Chunking. Defaults = Phase 3 winner (docs/experiments/phase3-chunking.md):
    # structured parent-child, 160-token children embedded, 800-token parents returned.
    # Phase 1 baseline: chunk_strategy=fixed, chunk_size=500, chunk_overlap=50, parent_chunk_size=0.
    # chunk_size must stay below the embedding model's max sequence length (512 for bge-small).
    chunk_size: int = Field(default=160, ge=8, le=510)
    chunk_overlap: int = Field(default=20, ge=0)
    clean_text: bool = True  # fix ligatures, hyphenation, layout residue, broken paragraphs
    chunk_strategy: Literal["fixed", "structured"] = "structured"
    contextual_headers: bool = False  # embed "Book › Chapter › Section" too (no measurable gain)
    # Parent-child: >0 means chunk_size is the (embedded) child size and parents of this
    # many tokens are what the LLM sees. Requires chunk_strategy="structured".
    parent_chunk_size: int = Field(default=800, ge=0, le=2000)

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})")
        if self.parent_chunk_size:
            if self.chunk_strategy != "structured":
                raise ValueError("parent_chunk_size requires chunk_strategy='structured'")
            if self.parent_chunk_size <= self.chunk_size:
                raise ValueError("parent_chunk_size must be larger than chunk_size")
        return self

    @property
    def chunks_path(self) -> Path:
        """Plain copy of the indexed chunks, one file per collection."""
        return self.data_dir / "chunks" / f"{self.collection}.jsonl"

    # Keyword index: store a BM25 sparse vector per chunk next to the dense one (cheap; needed
    # for retrieval_mode sparse/hybrid).
    index_sparse: bool = True

    # Retrieval (Phase 4, docs/experiments/phase4-retrieval.md)
    top_k: int = Field(default=5, ge=1, le=50)
    # Defaults = Phase 4 winner: hybrid first stage, MiniLM cross-encoder rerank of 50 candidates.
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"  # hybrid = dense + BM25 via RRF
    retrieval_candidates: int = Field(default=50, ge=1, le=500)  # per list, before fusion/rerank
    rrf_k: int = Field(default=60, ge=1)
    # Cross-encoder reranker over the candidates; "" disables reranking. On a GPU with 8 GB+,
    # BAAI/bge-reranker-v2-m3 is more accurate (needs ~2.6 GB; too big next to the LLM on 4 GB).
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_max_length: int = Field(default=512, ge=32, le=8192)
    rerank_batch_size: int = Field(default=32, ge=1)
    max_per_book: int = Field(default=0, ge=0)  # diversity cap in the top k; 0 = off
    expand_abbreviations: bool = False  # "MI" -> "MI (myocardial infarction)" before searching

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
