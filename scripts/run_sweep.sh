#!/bin/bash
# Run the E4 cheap-experiment matrix on the pod.
# Each run writes its own raw_*.jsonl file (the runner uses
# raw_{cond}_k{k}_seed{seed}.jsonl, but doesn't include ar_model in the
# filename — so we move the output after each run to avoid overwrites).
#
# Usage on pod (background, survives SSH drops):
#   cd /workspace/sfumato && nohup bash scripts/run_sweep.sh > sweep.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
LOG="e4/results/sweep_progress.log"
mkdir -p e4/results

run() {
    local cond=$1 k=$2 n=$3 seed=$4 ar=$5 out=$6
    echo "[$(date +%H:%M:%S)] running cond=$cond k=$k n=$n seed=$seed ar=$ar" | tee -a "$LOG"
    CONDITION="$cond" K_STEPS="$k" N_PROBLEMS="$n" SEED="$seed" \
        AR_MODEL="$ar" DIFF_MODEL="GSAI-ML/LLaDA-8B-Instruct" \
        MOCK_MODELS=0 HF_HOME=/workspace/.hf_cache \
        python3 e4/runner.py 2>&1 | tail -3 | tee -a "$LOG"
    if [ -f "e4/results/raw_${cond}_k${k}_seed${seed}.jsonl" ]; then
        mv "e4/results/raw_${cond}_k${k}_seed${seed}.jsonl" "$out"
        echo "  -> $out" | tee -a "$LOG"
    fi
}

QWEN05="Qwen/Qwen2.5-0.5B-Instruct"

# Prompt-format ablations at k=64, Qwen-0.5B planner where applicable
run c2hint   64 50 0 "$QWEN05" "e4/results/raw_c2hint_k64_seed0.jsonl"
run c2empty  64 50 0 "$QWEN05" "e4/results/raw_c2empty_k64_seed0.jsonl"
run crev     64 50 0 "$QWEN05" "e4/results/raw_crev_k64_seed0.jsonl"

# K-sweep on C2 (we already have k=64) and C3p
for k in 16 32 128; do
    run c2  "$k" 50 0 "$QWEN05" "e4/results/raw_c2_k${k}_seed0.jsonl"
    run c3p "$k" 50 0 "$QWEN05" "e4/results/raw_c3p_k${k}_seed0_qwen05.jsonl"
done

# C2 multi-seed at k=64 (we have seed=0, add 1, 2)
for seed in 1 2; do
    run c2 64 50 "$seed" "$QWEN05" "e4/results/raw_c2_k64_seed${seed}.jsonl"
done

echo "[$(date +%H:%M:%S)] DONE" | tee -a "$LOG"
