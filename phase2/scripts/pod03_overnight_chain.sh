#!/usr/bin/env bash
# Pod-03 overnight chain: rich-substrate → process-RM train → 2nd rich-substrate (problems 100-199) → re-train.
#
# Assumes pod-03 is already running phase2/spikes/option3-process-reward/make_rich_substrate.py
# for problems 0..99 (controlled by N_PROBLEMS=100). This script picks up after that.
#
# Usage (run inside SSH session):
#   bash phase2/scripts/pod03_overnight_chain.sh
set -u
cd /workspace/sfumato || exit 1
source .hf_env 2>/dev/null || true

PY=/workspace/sfumato/.venv/bin/python
LOGDIR=phase2/scripts/_pod03_logs
mkdir -p "$LOGDIR"

PHASE1_SUBSTRATE=phase2/spikes/option3-process-reward/rich_substrate.jsonl
PHASE2_SUBSTRATE=phase2/spikes/option3-process-reward/rich_substrate_p100_199.jsonl

echo "================================================================"
echo "[chain] $(date -u +%FT%TZ) — waiting for rich substrate (N=100, problems 0..99)"
echo "================================================================"
# Poll until either (a) the substrate generator has exited AND a recent jsonl
# exists, or (b) we time out at 3h. The make_rich_substrate.py wrapper exits
# when done; we look for absence of the python process.
deadline=$(( $(date +%s) + 10800 ))
while true; do
  if ! pgrep -f "make_rich_substrate.py" >/dev/null 2>&1; then
    echo "[chain] make_rich_substrate.py no longer running"
    break
  fi
  now=$(date +%s)
  if (( now > deadline )); then
    echo "[chain] WARN — 3h deadline hit waiting for substrate; proceeding anyway"
    break
  fi
  sleep 60
done

if [[ ! -s "$PHASE1_SUBSTRATE" ]]; then
  echo "[chain] FATAL — $PHASE1_SUBSTRATE missing/empty; aborting"
  exit 1
fi
echo "[chain] phase-1 substrate ready: $(wc -l < $PHASE1_SUBSTRATE) records"

# ---------------- step 1: train option-3 process-RM on phase-1 substrate ----------------
echo "================================================================"
echo "[chain] $(date -u +%FT%TZ) — training process-RM verifier (N=100)"
echo "================================================================"
"$PY" phase2/spikes/option3-process-reward/train_process_verifier.py 2>&1 \
  | tee "$LOGDIR/01_process_rm_n100.log"
cp -v phase2/spikes/option3-process-reward/option3_results.json \
       phase2/spikes/option3-process-reward/option3_results_n100.json 2>/dev/null || true

# ---------------- step 2: phase-2 rich substrate, problems 100..199 ----------------
echo "================================================================"
echo "[chain] $(date -u +%FT%TZ) — phase-2 rich substrate (problems 100..199)"
echo "================================================================"
# Save phase-1 substrate aside so phase-2 doesn't overwrite it.
cp -v "$PHASE1_SUBSTRATE" "${PHASE1_SUBSTRATE}.p0_99" 2>/dev/null || true
PROBLEM_OFFSET=100 N_PROBLEMS=100 BRANCHES=5 \
  "$PY" phase2/spikes/option3-process-reward/make_rich_substrate.py 2>&1 \
  | tee "$LOGDIR/02_substrate_p100_199.log"
mv -v "$PHASE1_SUBSTRATE" "$PHASE2_SUBSTRATE" 2>/dev/null || true
mv -v "${PHASE1_SUBSTRATE}.p0_99" "$PHASE1_SUBSTRATE" 2>/dev/null || true

# ---------------- step 3: combine + re-train on N=200 ----------------
echo "================================================================"
echo "[chain] $(date -u +%FT%TZ) — combine substrates + retrain on N=200"
echo "================================================================"
COMBINED=phase2/spikes/option3-process-reward/rich_substrate_n200.jsonl
cat "$PHASE1_SUBSTRATE" "$PHASE2_SUBSTRATE" > "$COMBINED"
echo "[chain] combined: $(wc -l < $COMBINED) records"
# point the trainer at the combined substrate via env var (script reads RICH_PATH if set)
RICH_PATH="$COMBINED" "$PY" phase2/spikes/option3-process-reward/train_process_verifier.py 2>&1 \
  | tee "$LOGDIR/03_process_rm_n200.log"
cp -v phase2/spikes/option3-process-reward/option3_results.json \
       phase2/spikes/option3-process-reward/option3_results_n200.json 2>/dev/null || true

echo "================================================================"
echo "[chain] $(date -u +%FT%TZ) — DONE"
echo "================================================================"
ls -la phase2/spikes/option3-process-reward/option3_results_n*.json 2>/dev/null
ls -la phase2/spikes/option3-process-reward/rich_substrate*.jsonl 2>/dev/null
