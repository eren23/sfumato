#!/bin/bash
# Pull E3a/E3b/E3c results from pods to local e5/results/
# Usage: ./pull_e3_results.sh [pod1_port pod1_host pod2_port pod2_host]
set -u
POD1_PORT=${1:-22079}
POD1_HOST=${2:-194.68.245.47}
POD2_PORT=${3:-22199}
POD2_HOST=${4:-69.30.85.103}

KEY="$HOME/.ssh/id_ed25519_runpod"
LOCAL_BASE="/Users/eren/Documents/ai/sfumato/e5/results"
REMOTE_BASE="/workspace/sfumato/e5/results"

pull() {
  local port=$1
  local host=$2
  local dir=$3
  echo "Pulling $dir from $host:$port..."
  mkdir -p "$LOCAL_BASE/$dir"
  rsync -avz --exclude='*/model.pt' \
    -e "ssh -i $KEY -p $port -o StrictHostKeyChecking=no" \
    "root@$host:$REMOTE_BASE/$dir/" "$LOCAL_BASE/$dir/" 2>&1 | tail -5
}

pull "$POD1_PORT" "$POD1_HOST" "e3a_probe5_n8"
pull "$POD2_PORT" "$POD2_HOST" "e3b_multiscale_d2"
pull "$POD2_PORT" "$POD2_HOST" "e3c_d3_crossover"
echo "DONE"
