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
```

## Evaluation
```bash
uv run medrag eval fetch-medqa              # MedQA USMLE test set (CC-BY-4.0) -> data/medqa/
uv run medrag eval build-synth --n 300      # synthetic retrieval set -> data/eval/ (one-off, ~30 min)

uv run medrag eval run                      # all suites -> eval/runs/<timestamp>-<model>/report.md
uv run medrag eval run -s smoke             # quick check (26 hand-written questions)
uv run medrag eval run -s medqa -m llama3.2:3b -m qwen3:4b   # compare local models
uv run medrag eval compare                  # key metrics of recent runs side by side

# Calibrate the automatic faithfulness check against your own judgement
uv run medrag eval label --n 50             # y/n per answer sentence (resumable)
uv run medrag eval calibrate --nli-model cross-encoder/nli-deberta-v3-base \
                             --nli-model cross-encoder/nli-deberta-v3-large
```

| Suite | What it measures |
|---|---|
| `retrieval` | hit@1/3/5/10, MRR@10, nDCG@10, right-book@5 on synthetic questions. Gold = exact evidence quote, so it stays valid when chunking changes |
| `generation` | faithfulness and citation precision (local NLI model), answer recall, false-refusal rate, TTFT, tokens/s |
| `smoke` | keyword pass rate, refusal on out-of-scope questions |
| `medqa` | USMLE accuracy closed-book vs. with retrieval, with 95% confidence intervals |

Generated datasets live in `data/` (gitignored): the synthetic set quotes the copyrighted textbooks.

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
