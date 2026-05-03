#!/usr/bin/env bash
# Verifier hunt: sweep candidate Qwen models on rich-substrate branches.
# Run on a pod with sufficient GPU memory. Each model writes to a separate JSON.
#
# Usage:
#   bash phase2/spikes/verifier-aggregation/run_verifier_hunt.sh
#
# 4090 24GB fits: Qwen3.6-27B 4-bit, Qwen3-Embedding-8B 4-bit
# A6000 48GB also fits: Qwen2.5-72B-Instruct 4-bit
set -u
cd /workspace/sfumato || exit 1
source .hf_env 2>/dev/null || true

PY=/workspace/sfumato/.venv/bin/python
SCRIPT=phase2/spikes/verifier-aggregation/train_verifier_option2.py
RES=phase2/spikes/verifier-aggregation/option2_results.json
LOGDIR=phase2/spikes/verifier-aggregation/_hunt_logs
mkdir -p "$LOGDIR"

declare -a MODELS=(
  "Qwen/Qwen3-Embedding-8B"
  "Qwen/Qwen3.6-27B"
  "Qwen/Qwen2.5-72B-Instruct"   # only on A6000+; will OOM on 4090
)

declare -a TAGS=(
  "qwen3emb8b"
  "qwen36-27b"
  "qwen25-72b-instruct"
)

for i in "${!MODELS[@]}"; do
  M="${MODELS[$i]}"
  T="${TAGS[$i]}"
  OUT="phase2/spikes/verifier-aggregation/option2_results_${T}.json"
  LOG="$LOGDIR/${T}.log"
  if [[ -f "$OUT" ]]; then
    echo "[hunt] skip $T (exists: $OUT)"; continue
  fi
  echo "============================================================"
  echo "[hunt] model: $M  tag: $T"
  echo "============================================================"
  "$PY" "$SCRIPT" --model-id "$M" --load-in-4bit --batch-size 4 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 && -f "$RES" ]]; then
    mv "$RES" "$OUT"
    echo "[hunt] done $T → $OUT"
  else
    echo "[hunt] FAIL $T (rc=$rc); see $LOG"
  fi
done

echo "============================================================"
echo "[hunt] all results:"
ls -la phase2/spikes/verifier-aggregation/option2_results_*.json 2>/dev/null
