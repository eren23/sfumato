#!/usr/bin/env bash
# Pod-side runner: ship the remaining 9 diff-mode SAEs (block_6..9, 11..15)
# and the E1 span-vs-token 60M training pair. Sequential because one GPU.
#
# Required env (typically set by the pod launcher / .env file):
#   WANDB_API_KEY, WANDB_PROJECT (=sfumato-e5), HUGGINGFACE_HUB_TOKEN
#
# Expected wall on a single A40:
#   - 9 diff SAEs x ~10 min each = ~90 min
#   - 2 E1 trainings (60M composite, 3000 steps) x ~15 min each = ~30 min
#   - 2 infill evals = ~5 min each = ~10 min
#   Total: ~2-2.5 hr, ~$1 on A40 at $0.44/hr.

set -euo pipefail

# Sfumato repo workspace on the pod
REPO_ROOT=${REPO_ROOT:-/workspace/sfumato}
cd "$REPO_ROOT"

# Need a HF tokens cache for SAE training. The pod's HF_HOME is set, but the
# 200M FineWeb cache is local-only. Re-tokenize on first use.
TOK_CACHE="$HOME/.cache/sfumato_e5"
mkdir -p "$TOK_CACHE"

if [[ ! -f "$TOK_CACHE/fineweb_gpt2_200000000.npy" ]]; then
    echo "[pod] tokens cache not found; tokenizing FineWeb-Edu (~10-20 min)..."
    python3 -u -c "
from e5.data import load_fineweb_tokens
import os
t = load_fineweb_tokens(n_tokens=200_000_000)
print(f'[pod] tokens ready: {len(t):,}')
"
fi

# F10 ckpt is large; pull from HF if not already present.
# HF repo layout: eren23/sfumato-composite-ckpts/f10_mixed/model_slim_final.pt
# Local layout (matches train_saes.py default): e5/results/f10_mixed/composite/model_slim_final.pt
export F10_LOCAL="$REPO_ROOT/e5/results/f10_mixed/composite/model_slim_final.pt"
mkdir -p "$REPO_ROOT/e5/results/f10_mixed/composite"
if [[ ! -f "$F10_LOCAL" ]]; then
    echo "[pod] F10 ckpt not found; pulling from HF eren23/sfumato-composite-ckpts..."
    F10_LOCAL="$F10_LOCAL" python3 -u -c "
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='eren23/sfumato-composite-ckpts',
                    filename='f10_mixed/model_slim_final.pt')
dst = os.environ['F10_LOCAL']
shutil.copy(p, dst)
print(f'[pod] F10 -> {dst}')
" || echo "[pod] WARN: F10 pull failed; SAE training will fail"
fi

echo "[pod] === Diff-mode SAEs (blocks 6..9, 11..15) ==="
export TOKENS_PATH="$TOK_CACHE/fineweb_gpt2_200000000.npy"
export DEVICE=cuda
export STEPS=${SAE_STEPS:-4000}
export BATCH=${SAE_BATCH:-8}
export T=${SAE_T:-256}

for L in 6 7 8 9 11 12 13 14 15; do
    OUT_DIR="$REPO_ROOT/e5/interp/saes/block_${L}_diff"
    if [[ -f "$OUT_DIR/sae.pt" ]]; then
        echo "[pod] block.${L}.diff already exists, skipping"
        continue
    fi
    echo "[pod] block.${L}.diff -- start $(date -u +%FT%TZ)"
    HOOKPOINT="block.${L}.diff" \
        HF_PUSH_REPO=${SAE_HF_PUSH_REPO:-} \
        python3 -u -m e5.interp.train_saes
    echo "[pod] block.${L}.diff -- done $(date -u +%FT%TZ)"
done

echo "[pod] === E1 span-vs-token 60M composite ==="
# Two identically-seeded composites that differ only in mask sampling mode.
export VARIANT=composite
export SEED=${E1_SEED:-201}
export MAX_STEPS=${E1_MAX_STEPS:-3000}
export BATCH_SIZE=8
export BLOCK_SIZE=256
export D_MODEL=512
export N_LAYERS=8
export N_HEADS=8
export EVAL_EVERY=500
export N_EVAL=50

for MM in uniform span_uniform; do
    OUT_DIR="$REPO_ROOT/e5/results/e1_span_vs_token/$MM"
    if [[ -f "$OUT_DIR/model.pt" ]]; then
        echo "[pod] e1/$MM already trained, skipping"
        continue
    fi
    echo "[pod] e1/$MM -- start $(date -u +%FT%TZ)"
    OUT_DIR="$OUT_DIR" MASK_MODE="$MM" python3 -u -m e5.train
    echo "[pod] e1/$MM -- trained $(date -u +%FT%TZ)"
done

echo "[pod] === E1 infill evaluation ==="
for MM in uniform span_uniform; do
    CKPT="$REPO_ROOT/e5/results/e1_span_vs_token/$MM/model.pt"
    if [[ -f "$CKPT" ]]; then
        OUT_JSON="$REPO_ROOT/e5/results/e1_span_vs_token/$MM/probe_infill.json"
        if [[ ! -f "$OUT_JSON" ]]; then
            CKPT="$CKPT" OUT="$OUT_JSON" N_EVAL=50 \
                python3 -u -m e5.scripts.probe_infill || echo "[pod] WARN: infill eval $MM failed"
        fi
    fi
done

echo "[pod] === all remaining work done $(date -u +%FT%TZ) ==="
