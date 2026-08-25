#!/bin/bash
# Local watcher for FlowGRPO+PickScore run. Fixes vs prior babysit:
#   1. Runs eval ON THE BOX FIRST, then pulls just the eval JSON + endpoint
#      checkpoints. Previous version rsynced the full flow_grpo logs tree
#      (15GB of checkpoint shards at 0.75 MB/s) BEFORE the eval — wasted ~$5
#      of idle-GPU time on a 4.5h rsync.
#   2. No-op runlogs rsync removed.
#
# Usage: bash scripts/cloud/babysit_flowgrpo_pickscore.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

set -a; source .env; set +a
: "${LAMBDA_API_KEY:?LAMBDA_API_KEY missing}"
: "${HF_TOKEN:?HF_TOKEN missing}"
IP="$(cat .lambda_ip)"
INST="$(cat .lambda_instance)"

SSH="ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"

mkdir -p logs outputs/checkpoints/flowgrpo_pickscore
LOG=logs/babysit_pickscore.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== babysit_pickscore started: ip=$IP instance=$INST ==="

DEADLINE=$(( $(date +%s) + 6*3600 ))
DONE_SENTINEL="=== FlowGRPO+PickScore run done ==="

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 90
    TAIL=$($SSH "ubuntu@$IP" 'tail -10 ~/haotian_research-1/logs/flowgrpo_pickscore_runner.log 2>/dev/null' 2>/dev/null || true)
    if echo "$TAIL" | grep -q "$DONE_SENTINEL"; then
        log "Runner finished."
        break
    fi
    HEART=$($SSH "ubuntu@$IP" 'tail -1 ~/haotian_research-1/logs/flowgrpo_pickscore_runner.log 2>/dev/null | tr -d "\r" | head -c 100; echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' 2>/dev/null || echo "ssh unreachable")
    log "heartbeat: $(echo "$HEART" | tr '\n' ' | ')"
done

# Run the offline CLIP eval on the box FIRST (uses the already-cached SD3.5-M
# weights + checkpoints already in /home/ubuntu/...).
log "Running offline PickScore eval on the box..."
$SSH "ubuntu@$IP" "HF_TOKEN=$HF_TOKEN bash ~/haotian_research-1/scripts/cloud/run_flowgrpo_pickscore_eval.sh" 2>&1 | tee -a "$LOG" || log "WARN: eval failed"

# Pull the eval JSON.
log "Pulling eval result..."
rsync -az -e "ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no" \
    "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/flowgrpo_pickscore_eval.json" \
    ./outputs/ 2>&1 | tee -a "$LOG" || log "WARN: eval rsync failed"

# Pull ONLY the endpoint checkpoints the eval used (not all 300+).
ENDPOINTS=$(python3 -c "
import json
d = json.load(open('outputs/flowgrpo_pickscore_eval.json'))
print(' '.join(k for k in d['rewards'].keys() if k != 'no_lora'))
" 2>/dev/null || echo "")
if [ -n "$ENDPOINTS" ]; then
    log "Pulling endpoint checkpoints: $ENDPOINTS"
    for ep in $ENDPOINTS; do
        rsync -az -e "ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no" \
            "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/checkpoints/flowgrpo_pickscore/$ep/" \
            "./outputs/checkpoints/flowgrpo_pickscore/$ep/" 2>&1 | tee -a "$LOG" || log "WARN: $ep rsync failed"
    done
fi

# Pull the training log (small) for completeness.
log "Pulling training log..."
rsync -az -e "ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no" \
    "ubuntu@$IP:/home/ubuntu/haotian_research-1/logs/" ./logs/ 2>&1 | tee -a "$LOG" || true

# Terminate.
log "Terminating Lambda instance $INST..."
RESP=$(curl -s -u "$LAMBDA_API_KEY:" \
    -H "Content-Type: application/json" \
    -d "{\"instance_ids\":[\"$INST\"]}" \
    https://cloud.lambdalabs.com/api/v1/instance-operations/terminate)
echo "$RESP" | tee -a "$LOG"
rm -f .lambda_instance .lambda_ip
log "=== babysit_pickscore done ==="
