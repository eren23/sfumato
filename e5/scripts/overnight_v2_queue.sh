#!/bin/bash
# Sfumato overnight v2 — runs F1/F2/F3 sequentially on a single pod.
#
# Usage on pod: nohup bash overnight_v2_queue.sh > e5/logs/overnight_v2.log 2>&1 &
#
# Total expected wall: ~4 hours on an A40.
# Results land in e5/results/f{1,2,3}_*/

set -u
cd /workspace/sfumato || exit 1

LOG=e5/logs/overnight_v2.log
mkdir -p e5/logs e5/results

echo "===== Overnight v2 start $(date) =====" >> $LOG

echo "===== F1 FineWeb composite ($(date)) =====" >> $LOG
TOKENS_LIST=5000000,10000000,50000000 SEEDS=110,111,112 MAX_STEPS=3000 OUT_NAME=f1_fineweb \
  python3 -u e5/scripts/f1_fineweb_composite.py >> $LOG 2>&1

echo "===== F2 alpha sweep ($(date)) =====" >> $LOG
ALPHAS=30,50,70 SEEDS=120,121,122 MAX_STEPS=3000 OUT_NAME=f2_alpha_sweep \
  python3 -u e5/scripts/f2_alpha_sweep.py >> $LOG 2>&1

echo "===== F3 AR compute control ($(date)) =====" >> $LOG
SCALES=60M,120M,200M SEEDS=130,131,132 MAX_STEPS=6000 OUT_NAME=f3_ar_compute_control \
  python3 -u e5/scripts/f3_ar_compute_control.py >> $LOG 2>&1

echo "===== Overnight v2 DONE $(date) =====" >> $LOG
