#!/usr/bin/env bash
# Phase 3 chunking experiment: one Qdrant collection per configuration, then the
# retrieval suite on each. Results: `uv run medrag eval compare`.
# Takes ~6 min per configuration on an RTX A500 (mostly embedding).
set -euo pipefail
cd "$(dirname "$0")/.."

# name | env overrides
CONFIGS=(
  "medrag_chunks|"  # Phase 1 baseline (already indexed): raw text, fixed 500/50
  "p3_fixed500_clean|MEDRAG_CLEAN_TEXT=true"
  "p3_struct500|MEDRAG_CHUNK_STRATEGY=structured"
  "p3_struct500_ctx|MEDRAG_CHUNK_STRATEGY=structured MEDRAG_CONTEXTUAL_HEADERS=true"
  "p3_struct256_ctx|MEDRAG_CHUNK_STRATEGY=structured MEDRAG_CONTEXTUAL_HEADERS=true MEDRAG_CHUNK_SIZE=256 MEDRAG_CHUNK_OVERLAP=32"
  "p3_pc160_800_ctx|MEDRAG_CHUNK_STRATEGY=structured MEDRAG_CONTEXTUAL_HEADERS=true MEDRAG_CHUNK_SIZE=160 MEDRAG_CHUNK_OVERLAP=20 MEDRAG_PARENT_CHUNK_SIZE=800"
)

for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"
  overrides="${entry#*|}"
  echo "=== ${name} (${overrides:-defaults})"
  if [[ "${name}" != "medrag_chunks" || ! -f "data/chunks/${name}.jsonl" ]]; then
    env MEDRAG_COLLECTION="${name}" ${overrides} uv run medrag ingest 2>&1 | grep -E "Indexed|Error|Traceback" || true
  fi
  env MEDRAG_COLLECTION="${name}" ${overrides} uv run medrag eval run -s retrieval 2>&1 | grep -E "hit@5 so far|Report" || true
done

uv run medrag eval compare --last "${#CONFIGS[@]}"
