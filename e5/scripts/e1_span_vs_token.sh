#!/usr/bin/env bash
# E1 — span-mask vs token-mask 60M proxy comparison.
#
# Trains two identically-seeded 60M composites that differ only in the
# diff-mode mask sampling primitive:
#   variant A: MASK_MODE=uniform        (Bernoulli per-token, current default)
#   variant B: MASK_MODE=span_uniform   (geometric-length span masking)
#
# Acceptance criterion (per situation report at
# /Users/eren/.claude/plans/what-is-next-for-splendid-crane.md):
#   >=5% improvement in diff-NLL on infill task for variant B vs A
#   AR-NLL within ±2% of baseline
#   then ship span masking as the new default.
#
# Local wall on Mac MPS: ~6 hr per variant at MAX_STEPS=3000 = ~12 hr total.
# Cloud: ~$3 on A40 each, ~30 min wall.
#
# Run with:
#   nohup bash e5/scripts/e1_span_vs_token.sh > e1.log 2>&1 &
# Then `tail -f e1.log`.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

export VARIANT=${VARIANT:-composite}
export SEED=${SEED:-201}
export MAX_STEPS=${MAX_STEPS:-3000}
export BATCH_SIZE=${BATCH_SIZE:-8}
export BLOCK_SIZE=${BLOCK_SIZE:-256}
export D_MODEL=${D_MODEL:-512}
export N_LAYERS=${N_LAYERS:-8}
export N_HEADS=${N_HEADS:-8}
export EVAL_EVERY=${EVAL_EVERY:-500}
export N_EVAL=${N_EVAL:-50}

echo "[e1] start at $(date -u +%FT%TZ)"

# ===== variant A: uniform token mask (control) =====
OUT_DIR=e5/results/e1_span_vs_token/uniform MASK_MODE=uniform \
    python3 -u -m e5.train

# ===== variant B: span-uniform mask =====
OUT_DIR=e5/results/e1_span_vs_token/span_uniform MASK_MODE=span_uniform \
    python3 -u -m e5.train

echo "[e1] training done at $(date -u +%FT%TZ)"

# ===== evaluation: diff-NLL on infill task =====
for V in uniform span_uniform; do
    CKPT="$REPO_ROOT/e5/results/e1_span_vs_token/$V/model.pt"
    if [[ -f "$CKPT" ]]; then
        echo "[e1] eval infill $V"
        OUT="$REPO_ROOT/e5/results/e1_span_vs_token/$V/probe_infill.json" \
            CKPT="$CKPT" N_EVAL=50 \
            python3 -u -m e5.scripts.probe_infill
    fi
done

echo "[e1] all done at $(date -u +%FT%TZ)"
