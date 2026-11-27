#!/usr/bin/env bash
# Phase 4 retrieval experiment on the Phase 3 index (structured parent-child 160->800, with BM25
# vectors): first-stage mode, rerankers, abbreviation expansion, per-book cap.
# Every configuration runs the retrieval suite only. Results: `uv run medrag eval compare`,
# significance: `uv run medrag eval paired <run A> <run B>`.
set -euo pipefail
cd "$(dirname "$0")/.."

MINILM=cross-encoder/ms-marco-MiniLM-L-6-v2
BGE=BAAI/bge-reranker-base
MEDCPT=ncbi/MedCPT-Cross-Encoder
M3=BAAI/bge-reranker-v2-m3

CONFIGS=(
  "MEDRAG_RETRIEVAL_MODE=dense"
  "MEDRAG_RETRIEVAL_MODE=sparse"
  "MEDRAG_RETRIEVAL_MODE=hybrid"
  "MEDRAG_RETRIEVAL_MODE=hybrid MEDRAG_RERANK_MODEL=${MINILM}"
  "MEDRAG_RETRIEVAL_MODE=hybrid MEDRAG_RERANK_MODEL=${MEDCPT}"
  "MEDRAG_RETRIEVAL_MODE=hybrid MEDRAG_RERANK_MODEL=${BGE}"
  "MEDRAG_RETRIEVAL_MODE=hybrid MEDRAG_RERANK_MODEL=${M3}"
  "MEDRAG_RETRIEVAL_MODE=dense MEDRAG_RERANK_MODEL=${BGE}"
  "MEDRAG_RETRIEVAL_MODE=sparse MEDRAG_RERANK_MODEL=${BGE}"
  "MEDRAG_RETRIEVAL_MODE=dense MEDRAG_RERANK_MODEL=${MINILM}"
  "MEDRAG_RETRIEVAL_MODE=sparse MEDRAG_RERANK_MODEL=${MINILM}"
  "MEDRAG_RETRIEVAL_MODE=hybrid MEDRAG_RERANK_MODEL=${MINILM} MEDRAG_RETRIEVAL_CANDIDATES=25"
  "MEDRAG_RETRIEVAL_MODE=hybrid MEDRAG_RERANK_MODEL=${MINILM} MEDRAG_RETRIEVAL_CANDIDATES=100"
  "MEDRAG_RETRIEVAL_MODE=hybrid MEDRAG_RERANK_MODEL=${MINILM} MEDRAG_EXPAND_ABBREVIATIONS=true"
  "MEDRAG_RETRIEVAL_MODE=hybrid MEDRAG_RERANK_MODEL=${MINILM} MEDRAG_MAX_PER_BOOK=2"
)

for overrides in "${CONFIGS[@]}"; do
  echo "=== ${overrides}"
  env ${overrides} uv run medrag eval run -s retrieval 2>&1 | grep -E "Report|Error|Traceback" || true
done

uv run medrag eval compare --last "${#CONFIGS[@]}"
