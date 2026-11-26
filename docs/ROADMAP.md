# MedRAG — Production RAG Roadmap

A staged plan for building a medical question-answering RAG system in Python over the
textbook corpus in `textbooks/en/`. Each phase produces something that runs end to end and
is measured against the same evaluation set, so every "advanced" technique has to earn
its place with numbers.

> **Safety framing.** This is an educational / clinical-reference assistant, not a
> diagnostic device. Answers must be grounded in retrieved text, cite their sources,
> refuse when evidence is missing, and carry a "not medical advice" notice. Patient-specific
> treatment decisions are out of scope.

---

## 0. Corpus facts (measured, not assumed)

| Property | Value | Consequence |
|---|---|---|
| Books | 17 English texts (Harrison, Robbins, Katzung, Gray, Nelson, DSM-5, First Aid, …) | Book + subject are natural metadata filters |
| Size | ~80 MB, ~12 M words (≈16 M tokens) | ~35–50 k chunks at 400–500 tokens: fits easily in one Qdrant node |
| Format | Plain text, one paragraph per line | Paragraph-aware chunking is cheap and effective |
| Noise | Thousands of short lines: section headings mixed with figure-caption fragments (`Fig. 1.1`, label debris) | Need a heading-vs-noise classifier in cleaning |
| Anomaly | `Surgery_Schwartz.txt` = 11 MB in **250 lines** | Must sentence-split very long lines before chunking |
| Biggest book | Harrison ≈ 28 % of corpus | Watch retrieval bias toward Harrison; consider per-book diversity in reranking |
| Licensing | Copyrighted textbooks (MedQA corpus) | **Never commit `textbooks/` to git**; it is gitignored |

Local hardware: 20 CPU cores, 15 GB RAM, RTX A500 (4 GB VRAM), Docker available.
That's enough for local embedding (small/base models in fp16), local reranking, and Qdrant.

---

## 1. Target architecture

```
                 ┌──────────── offline / batch ─────────────┐
textbooks/*.txt → clean → structure → chunk → enrich → embed ─┬→ Qdrant (dense + sparse, payload)
                                                              └→ chunk store (Parquet / Postgres)

                 ┌──────────────── online / per request ─────────────────────────────┐
user ─→ FastAPI ─→ guardrails (input) ─→ query understanding ─→ hybrid retrieve (top 50)
                    │                      (rewrite, expand,       │
                    │                       route by subject)      ▼
                    │                                         rerank (top 6–8)
                    │                                              │
                    │        semantic cache ◀──────────────────────┤
                    ▼                                              ▼
            response ◀─ guardrails (output) ◀─ Claude generation w/ citations (streamed)
                    │
                    └─→ traces, metrics, feedback → eval datasets
```

### Technology choices (with the reason for each)

| Layer | Choice | Why / alternatives |
|---|---|---|
| Language / packaging | Python 3.11+, `uv`, `pyproject.toml` | Fast installs, lockfile reproducibility |
| Config | `pydantic-settings` + `.env` | Typed config, 12-factor |
| Vector DB | **Qdrant** (Docker) | Native hybrid (dense + sparse/BM25) search, payload filters, quantization, snapshots. Alt: pgvector if you want one Postgres for everything |
| Dense embeddings | Start `BAAI/bge-small-en-v1.5` → evaluate **`ncbi/MedCPT`** (query/article encoders trained on PubMed search logs) and `bge-m3` | Medical-domain retrievers usually win on clinical queries; measure, don't assume |
| Sparse / lexical | BM25 (Qdrant sparse vectors via `fastembed`) or SPLADE | Drug names, eponyms, abbreviations (e.g. "SIADH", "Kawasaki") need exact-match recall |
| Reranker | `BAAI/bge-reranker-v2-m3` or `ncbi/MedCPT-Cross-Encoder` | Biggest single quality lever after hybrid search |
| LLM | **Claude Opus 5.5** (`claude-opus-5-5`) via the official `anthropic` SDK, streaming, adaptive thinking, `effort` tuned per route | Use a cheaper model (`claude-haiku-5-5`) only for auxiliary jobs (query rewriting, LLM-as-judge) if the eval shows it holds quality |
| API | FastAPI + SSE streaming, `uvicorn`/`gunicorn` | Async, OpenAPI docs |
| UI | Streamlit/Gradio (learning) → React (production) | |
| Cache / rate limits | Redis | Exact + semantic cache, per-user token buckets |
| Eval | Custom harness + `ragas`-style metrics + MedQA USMLE questions | MedQA questions were written against this exact corpus: a ready-made benchmark |
| Observability | OpenTelemetry → Langfuse or Phoenix; Prometheus + Grafana | Per-stage latency, token cost, retrieval hit rate |
| Deployment | Docker Compose → Kubernetes (optional) | |

---

## 2. Phased plan

Each phase lists **goals**, **build items**, **what you learn**, and an **exit criterion**.

### Phase 0: Foundations (½ day)
- Repo layout, `pyproject.toml`, `uv`, `ruff`, `mypy`, `pytest`, pre-commit.
- `.env.example`, `Settings` class, structured logging (`structlog`).
- `docker-compose.yml` with Qdrant (+ Redis later).
- **Exit:** `make test` and `docker compose up qdrant` both work.

### Phase 1: Naive RAG baseline (2–3 days)
Goal: the simplest thing that answers questions, so every later phase has a baseline.
1. **Load** the books → `Document(book, subject, text)`.
2. **Chunk** with a fixed size (e.g. 500 tokens, 50 overlap).
3. **Embed** with `bge-small-en-v1.5` (sentence-transformers, GPU, batch 64).
4. **Index** in Qdrant (one collection, cosine, payload = book/subject/chunk_id).
5. **Retrieve** top-k=5 dense.
6. **Generate** with Claude: system prompt says *answer only from the context, cite
   `[book §chunk]`, say "not found in the provided sources" otherwise*.
7. CLI: `medrag ask "What is the first-line treatment for absence seizures?"`
- **Learn:** the embedding → index → retrieve → prompt loop; where naive RAG fails.
- **Exit:** answers 20 hand-written questions; store them as `eval/smoke.jsonl`.

### Phase 2: Evaluation harness (2 days), built *before* optimizing
You can't improve what you don't measure.
- **Datasets**
  - `eval/smoke.jsonl`: 20–50 hand-written Q/A with the expected source book.
  - **MedQA (USMLE, 4-option)** dev split: accuracy of RAG vs. no-RAG (closed-book).
  - **Synthetic retrieval set**: for ~500 sampled chunks, have an LLM write a question that
    the chunk answers → gives (query, gold_chunk_id) pairs for retrieval metrics.
    Filter out questions that are too generic.
- **Metrics**
  - Retrieval: Recall@k, MRR, nDCG@10, gold-book hit rate.
  - Generation: faithfulness/groundedness (claims supported by context), answer relevance,
    citation precision (do cited chunks actually support the sentence?), refusal correctness.
  - Ops: p50/p95 latency per stage, tokens and $ per query.
- LLM-as-judge with a fixed rubric; calibrate it against ~50 human labels before trusting it.
- Results written to `eval/runs/<timestamp>/` and compared in a table/report.
- **Exit:** one command (`medrag eval --suite all`) produces a scorecard for the baseline.

### Phase 3: Data quality and smart chunking (3–4 days)
The highest return on effort in most RAG systems.
- **Cleaning:** unicode normalization, ligature fixes, de-hyphenation, drop figure/label debris
  (short lines with `Fig.`, isolated numbers, label fragments), collapse whitespace.
- **Long-line repair:** sentence-split `Surgery_Schwartz.txt`-style mega-lines (`pysbd` or
  spaCy/scispaCy sentencizer).
- **Structure recovery:** classify short lines as headings (Title Case, no terminal period,
  followed by a long paragraph) → build `book › chapter › section` breadcrumbs.
- **Chunking strategies to compare** (all evaluated in Phase 2 harness):
  - fixed tokens vs. paragraph-aware recursive vs. section-aware;
  - sizes 256 / 512 / 800 tokens;
  - **contextual chunk headers**: prepend `Book › Section` to each chunk before embedding;
  - **parent–child (small-to-big)**: embed ~200-token children, return the ~1,000-token parent.
- **Metadata:** book, subject (anatomy, pharmacology, …), section path, char offsets,
  content hash (for idempotent re-ingestion), and a tag for audience (Step 1 vs. clinical).
- **Exit:** a measurable Recall@10 improvement over Phase 1.

### Phase 4: Hybrid retrieval and reranking (3 days)
- Dense + sparse (BM25/SPLADE) in Qdrant, fused with **Reciprocal Rank Fusion**.
- Embedding model bake-off: bge-small vs. bge-m3 vs. MedCPT (query/article encoders).
- **Cross-encoder rerank:** retrieve 50 → rerank → keep 6–8.
- **Diversity:** MMR or per-book caps so Harrison doesn't crowd out specialty texts.
- **Metadata filtering/routing:** classify the query's subject → soft-boost matching books.
- Medical abbreviation and synonym expansion (e.g. "MI" ↔ "myocardial infarction") through a
  curated dictionary; UMLS/MeSH later if licensing allows.
- **Exit:** nDCG@10 and MedQA accuracy up vs. Phase 3; latency budget recorded.

### Phase 5: Advanced query understanding (3–4 days)
Add each technique only if the eval improves:
- **Conversational query rewriting:** turn multi-turn follow-ups ("what about in children?")
  into standalone queries.
- **Multi-query / decomposition:** split compound questions ("compare X and Y mechanisms
  and side effects") into sub-queries, retrieve per sub-query, merge.
- **HyDE:** embed a hypothetical answer; often helps short, vague queries.
- **Vignette handling:** USMLE-style clinical vignettes are long; extract key findings first,
  then retrieve.
- **Adaptive retrieval / routing:** skip retrieval for chit-chat; pick k by query complexity.
- **Exit:** gains on the multi-hop and vignette subsets of the eval.

### Phase 6: Grounded generation and citations (2–3 days)
- Pass retrieved chunks to Claude as `document` content blocks with
  `citations: {enabled: true}`, so the API returns sentence-level citations that point to
  exact source spans (no regex-parsed `[1]` markers).
- **Context engineering:** order chunks by relevance, dedupe near-duplicates, compress or
  trim to a token budget, keep the book/section header on each.
- **Prompt caching:** stable system prompt + instructions first, with a `cache_control`
  breakpoint; volatile retrieved context and the question after.
- **Answer contract:** structured sections (answer, key points, sources, confidence,
  disclaimer); explicit abstention when the context is insufficient.
- Streaming to the client (SSE) for perceived latency.
- Handle `stop_reason` (`refusal`, `max_tokens`) explicitly and enable server-side
  refusal fallbacks.
- **Exit:** faithfulness ≥ 0.9 and citation precision ≥ 0.9 on the eval.

### Phase 7: Safety and guardrails (2–3 days)
- **Input:** scope classifier (medical-education question vs. personal emergency vs.
  off-topic); emergency phrases → fixed "contact emergency services" response;
  prompt-injection checks on user input.
- **Output:** groundedness check (every claim backed by a citation), dosage/number
  verification against the cited text, mandatory disclaimer, no patient-specific prescribing.
- **PII:** detect and redact personal health information in logs (Presidio).
- **Corpus caveat:** textbook editions age; surface the source + edition so users can verify
  against current guidelines.
- Red-team set in `eval/redteam.jsonl` (unsafe requests, injection, out-of-scope).
- **Exit:** 100 % pass on the red-team suite.

### Phase 8: Productionization: API, performance, scale (4–5 days)
- **Service:** FastAPI with `/ask` (SSE streaming), `/feedback`, `/health`, `/ready`;
  async throughout (`AsyncAnthropic`, async Qdrant client); request IDs.
- **Ingestion pipeline:** idempotent, incremental (content hashes), versioned collections with
  **alias swap** for zero-downtime re-indexing; Qdrant snapshots for backup.
- **Performance:**
  - embedding + reranker served as a separate model service (batched; ONNX/TensorRT or
    Text-Embeddings-Inference container);
  - Qdrant HNSW tuning (`m`, `ef`), scalar/binary quantization, on-disk payloads;
  - **caching layers:** exact query cache → semantic cache (high-similarity threshold) →
    embedding cache → Claude prompt cache;
  - parallelize sparse + dense retrieval; set timeouts and circuit breakers per stage.
- **Reliability:** retries with backoff (SDK built-in), graceful degradation (reranker down →
  dense-only; LLM down → return top passages), rate limiting and per-user quotas.
- **Load testing:** Locust/k6; target e.g. p95 time-to-first-token < 2.5 s at N concurrent users.
- **Security:** API keys/OAuth, secrets from env or a secret manager, CORS, input size limits.
- **Exit:** load test passes the latency SLO; chaos tests show graceful degradation.

### Phase 9: Observability and continuous improvement (ongoing)
- OpenTelemetry traces per stage (rewrite → retrieve → rerank → generate), token and cost
  metrics, retrieval score distributions, abstention rate.
- Dashboards + alerts (latency, error rate, cost per day, cache hit rate).
- User feedback (👍/👎 + reason) → triaged into new eval cases.
- **CI gate:** the eval suite runs on every PR touching prompts/retrieval; block on regression.
- Prompt/version registry so every answer is traceable to (prompt v, index v, model).

### Phase 10: Advanced and experimental (pick by measured need)
- **Agentic RAG:** Claude decides when to search again (tool use: `search_textbooks(query,
  book_filter)`) for multi-hop questions; bounded iterations.
- **Corrective / self-reflective RAG (CRAG, Self-RAG):** grade retrieved docs, re-query on low
  relevance.
- **GraphRAG / knowledge graph:** entities (drugs, diseases, mechanisms) and relations for
  "what drugs interact with…" style questions.
- **Fine-tuning the retriever** on the synthetic (query, chunk) pairs from Phase 2.
- **Multilingual** (the MedQA corpus also ships Chinese texts).
- **Late interaction** retrieval (ColBERT) if reranking latency becomes a bottleneck.

---

## 3. Repository layout

```
medical-rag/
├── README.md
├── docs/ROADMAP.md            ← this file
├── pyproject.toml
├── .env.example
├── docker-compose.yml         (Phase 0)
├── textbooks/en/              (gitignored, local only)
├── data/                      (gitignored: cleaned text, chunks.parquet, caches)
├── src/medrag/
│   ├── config.py              settings
│   ├── ingest/                load · clean · structure · chunk · enrich
│   ├── index/                 embedders · qdrant store · sparse
│   ├── retrieve/              hybrid search · fusion · rerank · query rewrite
│   ├── generate/              prompts · claude client · citations
│   ├── guardrails/            input/output checks · PII
│   ├── api/                   FastAPI app · schemas · streaming
│   ├── eval/                  datasets · metrics · judge · runner
│   └── cli.py                 `medrag ingest | ask | eval`
├── eval/                      smoke.jsonl · redteam.jsonl · runs/
└── tests/
```

---

## 4. Milestones and checkpoints

| # | Milestone | Key metric to report |
|---|---|---|
| M1 | Phase 0–1 baseline answers questions | smoke-set pass rate |
| M2 | Phase 2 eval harness | baseline scorecard |
| M3 | Phase 3–4 better data + hybrid + rerank | Recall@10, nDCG@10, MedQA acc |
| M4 | Phase 5–7 query understanding, citations, guardrails | faithfulness, citation precision, red-team pass |
| M5 | Phase 8–9 production service | p95 TTFT, cost/query, error rate under load |
| M6 | Phase 10 experiments | delta vs. M5 on hard subsets |

## 5. Rules of thumb
1. Build the eval set before optimizing anything; change one variable per experiment.
2. Retrieval quality caps answer quality, so spend most effort on data, chunking and reranking.
3. Prefer "I don't know, here's what the sources say" over a confident wrong medical answer.
4. Log everything needed to reproduce an answer: query, chunks + scores, prompt version, model.
