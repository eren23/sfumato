#!/usr/bin/env bash
# Pod-07 retry: 3 fixable failures from hunt-v4 (gte/deepseek/qwq) — needed accelerate.
set -u
cd /workspace/sfumato || exit 1
source .env 2>/dev/null || true

PY=/workspace/sfumato/.venv/bin/python
SCRIPT=phase2/spikes/verifier-aggregation/train_verifier_option2.py
RES=phase2/spikes/verifier-aggregation/option2_results.json
LOGDIR=phase2/spikes/verifier-aggregation/_pod07_retry_logs
mkdir -p "$LOGDIR"

declare -a MODELS=(
  "Alibaba-NLP/gte-Qwen2-7B-instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  "Qwen/QwQ-32B"
)
declare -a TAGS=(
  "gte-qwen2-7b"
  "deepseek-r1-distill-qwen-32b"
  "qwq-32b"
)

for i in "${!MODELS[@]}"; do
  M="${MODELS[$i]}"
  T="${TAGS[$i]}"
  OUT="phase2/spikes/verifier-aggregation/option2_results_${T}.json"
  LOG="$LOGDIR/${T}.log"
  if [[ -f "$OUT" ]]; then
    echo "[pod07] skip $T (exists)"; continue
  fi
  echo "============================================================"
  echo "[pod07] $(date -u +%FT%TZ) — $M  → $T"
  echo "============================================================"
  "$PY" "$SCRIPT" --model-id "$M" --load-in-4bit --batch-size 4 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 && -f "$RES" ]]; then
    mv "$RES" "$OUT"
    echo "[pod07] DONE $T"
  else
    echo "[pod07] FAIL $T (rc=$rc)"
  fi
  CACHE_DIR="/workspace/sfumato/.hf_cache/models--${M//\//--}"
  [[ -d "$CACHE_DIR" ]] && { echo "[pod07] freeing $(du -sh "$CACHE_DIR" 2>/dev/null|cut -f1) — $CACHE_DIR"; rm -rf "$CACHE_DIR"; }
done

echo "============================================================"
echo "[pod07] DONE  results:"
ls -la phase2/spikes/verifier-aggregation/option2_results_*.json 2>/dev/null
