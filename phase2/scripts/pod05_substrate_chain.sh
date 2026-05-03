#!/usr/bin/env bash
# Pod-05 chain: rich-substrate N=100 (problems 0..99) → N=100 (problems 100..199)
#                  → train option-3 process verifier on combined N=200.
#
# Pod-05 starts EMPTY (no in-flight substrate run); this drives the whole thing
# from zero. Total wall-clock estimate: ~6h on RTX 4090.
set -u
cd /workspace/sfumato || exit 1
source .env 2>/dev/null || true

PY=/workspace/sfumato/.venv/bin/python
LOGDIR=phase2/scripts/_pod05_logs
mkdir -p "$LOGDIR"

PHASE1_SUBSTRATE=phase2/spikes/option3-process-reward/rich_substrate.jsonl
P0_99=phase2/spikes/option3-process-reward/rich_substrate_p0_99.jsonl
P100_199=phase2/spikes/option3-process-reward/rich_substrate_p100_199.jsonl
COMBINED=phase2/spikes/option3-process-reward/rich_substrate_n200.jsonl

# ---------------- step 1: substrate problems 0..99 ----------------
echo "================================================================"
echo "[chain05] $(date -u +%FT%TZ) — substrate p0..99 (N=100, b=5, ~3h)"
echo "================================================================"
N_PROBLEMS=100 PROBLEM_OFFSET=0 BRANCHES=5 \
  "$PY" phase2/spikes/option3-process-reward/make_rich_substrate.py 2>&1 \
  | tee "$LOGDIR/01_substrate_p0_99.log"
cp -v "$PHASE1_SUBSTRATE" "$P0_99" 2>/dev/null || true

# ---------------- step 2: train option-3 on N=100 ----------------
echo "================================================================"
echo "[chain05] $(date -u +%FT%TZ) — option-3 train on N=100"
echo "================================================================"
"$PY" phase2/spikes/option3-process-reward/train_process_verifier.py 2>&1 \
  | tee "$LOGDIR/02_option3_n100.log"
cp -v phase2/spikes/option3-process-reward/option3_results.json \
       phase2/spikes/option3-process-reward/option3_results_n100.json 2>/dev/null || true

# ---------------- step 3: substrate problems 100..199 ----------------
echo "================================================================"
echo "[chain05] $(date -u +%FT%TZ) — substrate p100..199 (N=100, b=5, ~3h)"
echo "================================================================"
N_PROBLEMS=100 PROBLEM_OFFSET=100 BRANCHES=5 \
  "$PY" phase2/spikes/option3-process-reward/make_rich_substrate.py 2>&1 \
  | tee "$LOGDIR/03_substrate_p100_199.log"
mv -v "$PHASE1_SUBSTRATE" "$P100_199" 2>/dev/null || true
mv -v "$P0_99" "$PHASE1_SUBSTRATE" 2>/dev/null || true

# ---------------- step 4: combine + retrain on N=200 ----------------
echo "================================================================"
echo "[chain05] $(date -u +%FT%TZ) — combine + retrain on N=200"
echo "================================================================"
cat "$PHASE1_SUBSTRATE" "$P100_199" > "$COMBINED"
echo "[chain05] combined: $(wc -l < $COMBINED) records"
RICH_PATH="$COMBINED" "$PY" phase2/spikes/option3-process-reward/train_process_verifier.py 2>&1 \
  | tee "$LOGDIR/04_option3_n200.log"
cp -v phase2/spikes/option3-process-reward/option3_results.json \
       phase2/spikes/option3-process-reward/option3_results_n200.json 2>/dev/null || true

echo "================================================================"
echo "[chain05] $(date -u +%FT%TZ) — DONE"
echo "================================================================"
ls -la phase2/spikes/option3-process-reward/option3_results_n*.json 2>/dev/null
ls -la phase2/spikes/option3-process-reward/rich_substrate*.jsonl 2>/dev/null
