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

## Quick start (as phases land)
```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env              # set the Ollama URL and model (no API keys)
docker compose up -d qdrant       # Phase 0
medrag doctor                     # checks Qdrant, Ollama, model, GPU
medrag ingest                     # Phase 1
medrag ask "First-line treatment for absence seizures?"
medrag eval --suite smoke         # Phase 2
```

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
