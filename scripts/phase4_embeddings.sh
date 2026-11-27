#!/usr/bin/env bash
# Phase 4 embedding bake-off: same chunks (sizes measured with the bge-small tokenizer), one
# collection per embedding model, retrieval suite in dense mode (pure embedding quality).
# The bge-small baseline is the default collection `medrag_pc160_800`.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="MEDRAG_CHUNK_TOKENIZER=BAAI/bge-small-en-v1.5 MEDRAG_RETRIEVAL_MODE=dense"
# name | env overrides
CONFIGS=(
  "p4_emb_bge_base|MEDRAG_EMBED_MODEL=BAAI/bge-base-en-v1.5"
  "p4_emb_medcpt|MEDRAG_EMBED_MODEL=ncbi/MedCPT-Article-Encoder MEDRAG_EMBED_QUERY_MODEL=ncbi/MedCPT-Query-Encoder MEDRAG_EMBED_POOLING=cls MEDRAG_EMBED_QUERY_PREFIX="
)

for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"
  overrides="${entry#*|}"
  echo "=== ${name} (${overrides})"
  env MEDRAG_COLLECTION="${name}" ${COMMON} ${overrides} uv run medrag ingest 2>&1 \
    | grep -E "Indexed|Error|Traceback" || true
  env MEDRAG_COLLECTION="${name}" ${COMMON} ${overrides} uv run medrag eval run -s retrieval 2>&1 \
    | grep -E "Report|Error|Traceback" || true
done

uv run medrag eval compare --last "${#CONFIGS[@]}"
