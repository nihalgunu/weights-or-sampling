#!/bin/bash
# Local watcher for the FlowGRPO+CLIP run. Polls remote runner, rsyncs results
# back, runs the offline CLIP eval on the box, then terminates the instance.
#
# Usage: bash scripts/cloud/babysit_flowgrpo_clip.sh
# Reads: .env (LAMBDA_API_KEY, HF_TOKEN), .lambda_ip, .lambda_instance.

set -euo pipefail
cd "$(dirname "$0")/../.."

set -a; source .env; set +a
: "${LAMBDA_API_KEY:?LAMBDA_API_KEY missing}"
: "${HF_TOKEN:?HF_TOKEN missing — needed for offline CLIP eval (SD3.5-M)}"
IP="$(cat .lambda_ip)"
INST="$(cat .lambda_instance)"

SSH="ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"

mkdir -p logs outputs/checkpoints/flowgrpo_clip
LOG=logs/babysit_clip.log

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== babysit_clip started: ip=$IP instance=$INST ==="

# 5h training cap + 1h grace for eval & artifact pull.
DEADLINE=$(( $(date +%s) + 6*3600 ))
DONE_SENTINEL="=== FlowGRPO+CLIP run done ==="

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 90
    TAIL=$($SSH "ubuntu@$IP" 'tail -10 ~/haotian_research-1/logs/flowgrpo_clip_runner.log 2>/dev/null' 2>/dev/null || true)
    if echo "$TAIL" | grep -q "$DONE_SENTINEL"; then
        log "Runner finished. Pulling artifacts..."
        break
    fi
    HEART=$($SSH "ubuntu@$IP" 'tail -1 ~/haotian_research-1/logs/flowgrpo_clip_runner.log 2>/dev/null | tr -d "\r" | head -c 100; echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' 2>/dev/null || echo "ssh unreachable")
    log "heartbeat: $(echo "$HEART" | tr '\n' ' | ')"
done

log "Rsyncing logs..."
rsync -az -e "ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no" \
    "ubuntu@$IP:~/haotian_research-1/logs/" ./logs/ 2>&1 | tee -a "$LOG" || log "WARN: log rsync failed"

log "Rsyncing checkpoints..."
rsync -az -e "ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no" \
    "ubuntu@$IP:~/haotian_research-1/outputs/checkpoints/flowgrpo_clip/" ./outputs/checkpoints/flowgrpo_clip/ 2>&1 | tee -a "$LOG" || log "WARN: ckpt rsync failed"

log "Rsyncing flow_grpo run logs..."
rsync -az -e "ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no" \
    "ubuntu@$IP:~/haotian_research-1/repos/flow_grpo/logs/clip_score/" ./outputs/flowgrpo_clip_runlogs/ 2>&1 | tee -a "$LOG" || log "WARN: runlogs rsync failed"

# --- Run the offline CLIP eval on the box (still has SD3.5-M cached) ---
log "Running offline CLIP eval on the box..."
$SSH "ubuntu@$IP" "HF_TOKEN=$HF_TOKEN bash ~/haotian_research-1/scripts/cloud/run_flowgrpo_clip_eval.sh" 2>&1 | tee -a "$LOG" || log "WARN: eval failed"

log "Pulling eval result..."
rsync -az -e "ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no" \
    "ubuntu@$IP:~/haotian_research-1/outputs/flowgrpo_clip_eval.json" ./outputs/ 2>&1 | tee -a "$LOG" || log "WARN: eval rsync failed"

# Terminate.
log "Terminating Lambda instance $INST..."
RESP=$(curl -s -u "$LAMBDA_API_KEY:" \
    -H "Content-Type: application/json" \
    -d "{\"instance_ids\":[\"$INST\"]}" \
    https://cloud.lambdalabs.com/api/v1/instance-operations/terminate)
echo "$RESP" | tee -a "$LOG"
rm -f .lambda_instance .lambda_ip

log "=== babysit_clip done ==="
