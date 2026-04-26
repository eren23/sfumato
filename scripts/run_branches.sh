#!/bin/bash
# Branch-style hybrid experiments: parallel-diffuse-then-converge.
# Run AFTER scripts/run_sweep.sh finishes.
#
# Usage on pod:
#   nohup bash scripts/run_branches.sh > sweep_branches.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
LOG="e4/results/branches_progress.log"
mkdir -p e4/results

# Pull WANDB credentials from /workspace/.wandb_env if present.
if [ -f /workspace/.wandb_env ]; then
    set -a
    source /workspace/.wandb_env
    set +a
fi

run() {
    local cond=$1 k=$2 n=$3 seed=$4 ar=$5 branches=$6 out=$7
    echo "[$(date +%H:%M:%S)] cond=$cond k=$k n=$n branches=$branches" | tee -a "$LOG"
    CONDITION="$cond" K_STEPS="$k" N_PROBLEMS="$n" SEED="$seed" \
        AR_MODEL="$ar" DIFF_MODEL="GSAI-ML/LLaDA-8B-Instruct" \
        BRANCHES="$branches" \
        MOCK_MODELS=0 HF_HOME=/workspace/.hf_cache \
        python3 e4/runner.py 2>&1 | tail -3 | tee -a "$LOG"
    if [ -f "e4/results/raw_${cond}_k${k}_seed${seed}.jsonl" ]; then
        mv "e4/results/raw_${cond}_k${k}_seed${seed}.jsonl" "$out"
        echo "  -> $out" | tee -a "$LOG"
    fi
}

QWEN05="Qwen/Qwen2.5-0.5B-Instruct"

# Self-consistency on diffusion (no AR): does parallel branching alone help?
run cmaj 64 50 0 "$QWEN05" 3 "e4/results/raw_cmaj_k64_seed0_b3.jsonl"
run cmaj 64 50 0 "$QWEN05" 5 "e4/results/raw_cmaj_k64_seed0_b5.jsonl"

# Parallel-diffuse + AR-merge (the inverse of C3 — branch-then-converge).
run cmerge 64 50 0 "$QWEN05" 3 "e4/results/raw_cmerge_k64_seed0_b3.jsonl"

echo "[$(date +%H:%M:%S)] DONE branches" | tee -a "$LOG"
