#!/usr/bin/env bash
# Phase 4 end-to-end check on 300 MedQA questions (the same seeded sample in every run):
# Phase 3 retrieval vs. the Phase 4 pipeline, with and without abbreviation expansion.
# The Phase 4 run also runs every other suite. ~30 min per run on a 4 GB GPU.
# Significance: `uv run medrag eval paired <run A> <run B> -m medqa_rag`.
set -euo pipefail
cd "$(dirname "$0")/.."

P3="MEDRAG_RETRIEVAL_MODE=dense MEDRAG_RERANK_MODEL="
P4="MEDRAG_RETRIEVAL_MODE=hybrid MEDRAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2"

env ${P3} uv run medrag eval run -s medqa --medqa-limit 300 2>&1 | grep -E "medqa 300|Report|Error|Traceback" || true
env ${P4} uv run medrag eval run --medqa-limit 300 2>&1 | grep -E "medqa 300|Report|Error|Traceback" || true
env ${P4} MEDRAG_EXPAND_ABBREVIATIONS=true uv run medrag eval run -s medqa --medqa-limit 300 2>&1 \
  | grep -E "medqa 300|Report|Error|Traceback" || true

uv run medrag eval compare --last 3
