#!/usr/bin/env bash
# Verifier hunt v4 — all night-2 blocked candidates, fresh transformers 4.55+.
# Requires: 200GB volume + transformers ≥4.55 (for qwen3_5 model_type).
#
# Usage (run inside pod SSH):
#   bash phase2/spikes/verifier-aggregation/run_verifier_hunt_v4.sh
set -u
cd /workspace/sfumato || exit 1
source .env 2>/dev/null || true

PY=/workspace/sfumato/.venv/bin/python
SCRIPT=phase2/spikes/verifier-aggregation/train_verifier_option2.py
RES=phase2/spikes/verifier-aggregation/option2_results.json
LOGDIR=phase2/spikes/verifier-aggregation/_hunt_v4_logs
mkdir -p "$LOGDIR"

# Models in size order (smallest first, so we get results quickly even if late ones OOM)
declare -a MODELS=(
  "Alibaba-NLP/gte-Qwen2-7B-instruct"           # ~14GB 4-bit; needs trust_remote_code
  "Qwen/Qwen3.6-27B"                             # ~14GB 4-bit; needs transformers ≥4.55 for qwen3_5
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"   # ~16GB 4-bit
  "Qwen/QwQ-32B"                                 # ~16GB 4-bit
  "Qwen/Qwen2.5-32B-Instruct"                   # ~16GB 4-bit
  "Qwen/Qwen2.5-72B-Instruct"                   # ~36GB 4-bit
  "Qwen/Qwen2.5-Math-72B-Instruct"              # ~36GB 4-bit
)

declare -a TAGS=(
  "gte-qwen2-7b"
  "qwen36-27b"
  "deepseek-r1-distill-qwen-32b"
  "qwq-32b"
  "qwen25-32b-instruct"
  "qwen25-72b-instruct"
  "qwen25-math-72b-instruct"
)

for i in "${!MODELS[@]}"; do
  M="${MODELS[$i]}"
  T="${TAGS[$i]}"
  OUT="phase2/spikes/verifier-aggregation/option2_results_${T}.json"
  LOG="$LOGDIR/${T}.log"
  if [[ -f "$OUT" ]]; then
    echo "[hunt-v4] skip $T (exists: $OUT)"; continue
  fi
  echo "============================================================"
  echo "[hunt-v4] $(date -u +%FT%TZ) — $M  → $T"
  echo "============================================================"
  df -h /workspace 2>/dev/null | tail -1
  "$PY" "$SCRIPT" --model-id "$M" --load-in-4bit --batch-size 4 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 && -f "$RES" ]]; then
    mv "$RES" "$OUT"
    echo "[hunt-v4] DONE $T → $OUT"
  else
    echo "[hunt-v4] FAIL $T (rc=$rc)"
  fi
  CACHE_DIR="/workspace/sfumato/.hf_cache/models--${M//\//--}"
  if [[ -d "$CACHE_DIR" ]]; then
    SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
    echo "[hunt-v4] freeing $SIZE — $CACHE_DIR"
    rm -rf "$CACHE_DIR"
  fi
done

echo "============================================================"
echo "[hunt-v4] $(date -u +%FT%TZ) — DONE  results:"
ls -la phase2/spikes/verifier-aggregation/option2_results_*.json 2>/dev/null
