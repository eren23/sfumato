#!/usr/bin/env bash
# Pod-07 overnight chain: extra-substrate p200..499 + 3 verifier variants.
#
# 1. Extend rich substrate from N=200 → N=500 (problems 200..499 = 300 new × 5 branches = 1500 records)
# 2. Train option-3 process verifier on N=500 combined
# 3. Train branch-pair contrastive verifier on N=500
# 4. Train step-level PRM on N=500
set -u
cd /workspace/sfumato || exit 1
source .env 2>/dev/null || true

PY=/workspace/sfumato/.venv/bin/python
LOGDIR=phase2/scripts/_pod07_logs
mkdir -p "$LOGDIR"

# Sub-substrates
PHASE1_SUBSTRATE=phase2/spikes/option3-process-reward/rich_substrate.jsonl  # transient working file
P0_99=phase2/spikes/option3-process-reward/rich_substrate_p0_99.jsonl       # imported
P100_199=phase2/spikes/option3-process-reward/rich_substrate_p100_199.jsonl # imported
P200_499=phase2/spikes/option3-process-reward/rich_substrate_p200_499.jsonl # to-build
N500=phase2/spikes/option3-process-reward/rich_substrate_n500.jsonl

# ---------------- step 1: substrate problems 200..499 ----------------
echo "================================================================"
echo "[chain07] $(date -u +%FT%TZ) — substrate p200..499 (N=300, b=5, ~3.5h)"
echo "================================================================"
N_PROBLEMS=300 PROBLEM_OFFSET=200 BRANCHES=5 \
  DEV_INDICES_PATH=/workspace/sfumato/e4/data/gsm8k_dev_500.json \
  "$PY" phase2/spikes/option3-process-reward/make_rich_substrate.py 2>&1 \
  | tee "$LOGDIR/01_substrate_p200_499.log"
mv -v "$PHASE1_SUBSTRATE" "$P200_499" 2>/dev/null || true

# ---------------- step 2: combine into N=500 ----------------
echo "================================================================"
echo "[chain07] $(date -u +%FT%TZ) — combine substrates into N=500"
echo "================================================================"
cat "$P0_99" "$P100_199" "$P200_499" > "$N500"
echo "[chain07] N=500 substrate: $(wc -l < $N500) records"

# ---------------- step 3: option-3 process MLP on N=500 ----------------
echo "================================================================"
echo "[chain07] $(date -u +%FT%TZ) — train option-3 process MLP on N=500"
echo "================================================================"
RICH_PATH="$N500" "$PY" phase2/spikes/option3-process-reward/train_process_verifier.py 2>&1 \
  | tee "$LOGDIR/02_option3_n500.log"
cp -v phase2/spikes/option3-process-reward/option3_results.json \
       phase2/spikes/option3-process-reward/option3_results_n500.json 2>/dev/null || true

# ---------------- step 4: branch-pair contrastive on N=500 ----------------
echo "================================================================"
echo "[chain07] $(date -u +%FT%TZ) — branch-pair contrastive on N=500"
echo "================================================================"
RICH_PATH="$N500" "$PY" phase2/spikes/option3-process-reward/train_branchpair_contrastive.py 2>&1 \
  | tee "$LOGDIR/03_branchpair_n500.log"

# ---------------- step 5: step-level PRM on N=500 ----------------
echo "================================================================"
echo "[chain07] $(date -u +%FT%TZ) — step-level PRM on N=500"
echo "================================================================"
RICH_PATH="$N500" "$PY" phase2/spikes/option3-process-reward/train_step_level_prm.py 2>&1 \
  | tee "$LOGDIR/04_step_prm_n500.log"

echo "================================================================"
echo "[chain07] $(date -u +%FT%TZ) — DONE"
echo "================================================================"
ls -la phase2/spikes/option3-process-reward/option3_*.json 2>/dev/null
