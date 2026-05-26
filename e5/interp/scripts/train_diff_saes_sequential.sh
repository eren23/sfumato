#!/usr/bin/env bash
# Sequentially train diff-mode SAEs for blocks 6-15 to make the
# SAE coverage symmetric with the existing AR-mode SAEs (currently 11
# AR + 1 diff).  Required before any "mode-specific" feature claim in
# the interp paper, per the situation report at
# /Users/eren/.claude/plans/what-is-next-for-splendid-crane.md (move 2).
#
# Wall time on Mac MPS: ~60 min per SAE x 10 = ~10 hr.
# Each SAE writes to e5/interp/saes/block_{L}_diff/sae.pt.
#
# Override env to change scope:
#   STEPS=8000     full-length training (default 4000 = ~30 min/SAE)
#   BATCH=8 T=256  (defaults; reduce if MPS OOMs)
#   START_L=6 END_L=15  layer range
#
# Run with:
#   nohup bash e5/interp/scripts/train_diff_saes_sequential.sh > diff_saes.log 2>&1 &
# Then `tail -f diff_saes.log` to monitor.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"

export TOKENS_PATH=${TOKENS_PATH:-$HOME/.cache/sfumato_e5/fineweb_gpt2_200000000.npy}
export DEVICE=${DEVICE:-mps}
export STEPS=${STEPS:-4000}
export BATCH=${BATCH:-8}
export T=${T:-256}

START_L=${START_L:-6}
END_L=${END_L:-15}

if [[ ! -f "$TOKENS_PATH" ]]; then
    echo "ERROR: TOKENS_PATH=$TOKENS_PATH not found" >&2
    exit 1
fi

echo "[diff_saes] start at $(date -u +%FT%TZ)"
echo "[diff_saes] tokens=$TOKENS_PATH device=$DEVICE steps=$STEPS"
echo "[diff_saes] layers ${START_L}..${END_L}"

for L in $(seq "$START_L" "$END_L"); do
    OUT_DIR="$REPO_ROOT/e5/interp/saes/block_${L}_diff"
    if [[ -f "$OUT_DIR/sae.pt" ]]; then
        echo "[diff_saes] block.${L}.diff -- already exists at $OUT_DIR/sae.pt, skipping"
        continue
    fi
    echo "[diff_saes] block.${L}.diff -- start $(date -u +%FT%TZ)"
    HOOKPOINT="block.${L}.diff" python3 -u -m e5.interp.train_saes
    echo "[diff_saes] block.${L}.diff -- done"
done

echo "[diff_saes] all done at $(date -u +%FT%TZ)"
