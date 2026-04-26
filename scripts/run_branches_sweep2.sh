#!/bin/bash
# Round 2 branches experiments: temperature sweep, N=10 branches,
# Qwen-AR self-consistency baseline.
# Run on pod via nohup.

set -u
cd "$(dirname "$0")/.."
LOG="e4/results/branches2_progress.log"
mkdir -p e4/results

if [ -f /workspace/.wandb_env ]; then
    set -a
    source /workspace/.wandb_env
    set +a
fi

run() {
    local cond=$1 k=$2 n=$3 seed=$4 ar=$5 branches=$6 temp=$7 out=$8
    echo "[$(date +%H:%M:%S)] cond=$cond k=$k n=$n branches=$branches temp=$temp" | tee -a "$LOG"
    CONDITION="$cond" K_STEPS="$k" N_PROBLEMS="$n" SEED="$seed" \
        AR_MODEL="$ar" DIFF_MODEL="GSAI-ML/LLaDA-8B-Instruct" \
        BRANCHES="$branches" TEMP="$temp" \
        MOCK_MODELS=0 HF_HOME=/workspace/.hf_cache \
        python3 e4/runner.py 2>&1 | tail -3 | tee -a "$LOG"
    if [ -f "e4/results/raw_${cond}_k${k}_seed${seed}.jsonl" ]; then
        mv "e4/results/raw_${cond}_k${k}_seed${seed}.jsonl" "$out"
        echo "  -> $out" | tee -a "$LOG"
    fi
}

QWEN05="Qwen/Qwen2.5-0.5B-Instruct"

# Temperature sweep on cmaj b=5 (already have temp=0.7=80%)
run cmaj 64 50 0 "$QWEN05" 5 0.3 "e4/results/raw_cmaj_k64_seed0_b5_t0.3.jsonl"
run cmaj 64 50 0 "$QWEN05" 5 1.0 "e4/results/raw_cmaj_k64_seed0_b5_t1.0.jsonl"

# Scale: N=10 branches at temp=0.7
run cmaj 64 50 0 "$QWEN05" 10 0.7 "e4/results/raw_cmaj_k64_seed0_b10_t0.7.jsonl"

# Qwen-AR self-consistency baseline: 5 Qwen runs with temperature, vote
# This is to test "is voting model-agnostic, or does diffusion specifically benefit?"
# Approximate via multiple seeds + a small temp on Qwen.
# (Not implemented as a condition — we synthesize from existing C1 multi-seed.
# Skipping for now; deferred to next round if signal motivates.)

echo "[$(date +%H:%M:%S)] DONE branches2" | tee -a "$LOG"
