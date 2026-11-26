# MedRAG

A production-grade Retrieval-Augmented Generation (RAG) assistant for medical education,
built step by step in Python, from a naive baseline up to hybrid retrieval, reranking,
grounded citations, guardrails and a scalable API.

**See [docs/ROADMAP.md](docs/ROADMAP.md) for the full phased plan.**

> ⚠️ Educational project. Not a medical device and not medical advice.

**Runs 100 % locally with open-source components. No cloud LLM API keys required.**

## Stack
Python 3.11 · **Ollama** (local open-weight LLMs) · Qdrant (hybrid dense + sparse) ·
sentence-transformers / MedCPT embeddings · cross-encoder reranking · FastAPI · Valkey ·
OpenTelemetry + Langfuse (self-hosted)

## Prerequisites
- [Ollama](https://ollama.com) running locally, with a model pulled, e.g.
  `ollama pull llama3.2:3b` (see the model sizing table in the roadmap)
- Docker (for Qdrant)
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)

## Data
The source corpus is a set of English medical textbooks (the MedQA textbook corpus) placed
locally under `textbooks/en/`. **The textbooks are copyrighted and are not included in this
repository** (`textbooks/` is gitignored). Obtain them separately and copy them in.

## Quick start
```bash
make install                      # .venv with Python 3.11 + all dependencies (uv)
cp .env.example .env              # set the Ollama URL and model (no API keys)
make up                           # start Qdrant
uv run medrag doctor              # checks Qdrant, Ollama, model, GPU, corpus

uv run medrag ingest              # chunk + embed + index all books (or: --book Anatomy_Gray)
uv run medrag ask "What is the first-line drug for absence seizures?" --show-context
uv run medrag smoke               # 22 smoke questions: answer + retrieval pass rates
```

Smoke results are written to `eval/runs/`. A full evaluation harness arrives in Phase 2.

## Status
- [ ] Phase 0: Foundations
- [ ] Phase 1: Naive RAG baseline
- [ ] Phase 2: Evaluation harness
- [ ] Phase 3: Cleaning and smart chunking
- [ ] Phase 4: Hybrid retrieval and reranking
- [ ] Phase 5: Query understanding
- [ ] Phase 6: Grounded generation and citations
- [ ] Phase 7: Safety and guardrails
- [ ] Phase 8: Production API and performance
- [ ] Phase 9: Observability and CI eval gates
- [ ] Phase 10: Advanced (agentic, GraphRAG, fine-tuning)
