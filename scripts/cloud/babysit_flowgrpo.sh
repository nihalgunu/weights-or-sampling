#!/bin/bash
# Local watcher: polls remote run_flowgrpo.sh, then rsyncs results back and
# terminates the Lambda instance. Run in the background and forget.
#
# Usage: bash scripts/cloud/babysit_flowgrpo.sh
# Reads: .env (LAMBDA_API_KEY), .lambda_ip, .lambda_instance.
# Writes: logs/babysit.log (status), logs/flowgrpo_runner.log (mirror of remote).

set -euo pipefail
cd "$(dirname "$0")/../.."

set -a; source .env; set +a
: "${LAMBDA_API_KEY:?LAMBDA_API_KEY missing}"
IP="$(cat .lambda_ip)"
INST="$(cat .lambda_instance)"

SSH="ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"
RSYNC="rsync -az -e \"ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no\""

mkdir -p logs outputs/checkpoints/flowgrpo
LOG=logs/babysit.log

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== babysit started: ip=$IP instance=$INST ==="
log "Polling remote runner every 90s for completion sentinel..."

# Poll until the runner script writes the done sentinel, or until SSH is dead
# for too long. Cap at 6 hours regardless (5h training + 1h grace).
DEADLINE=$(( $(date +%s) + 6*3600 ))
DONE_SENTINEL="=== FlowGRPO mini run done ==="

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 90
    # Pull a wide tail — the runner writes 4 lines after the sentinel, so
    # tail -3 misses it. Use || true so a transient SSH error doesn't kill us.
    TAIL=$($SSH "ubuntu@$IP" 'tail -10 ~/haotian_research-1/logs/flowgrpo_runner.log 2>/dev/null' 2>/dev/null || true)
    if echo "$TAIL" | grep -q "$DONE_SENTINEL"; then
        log "Runner finished. Pulling artifacts..."
        break
    fi
    # Light heartbeat — last log line + GPU util.
    HEART=$($SSH "ubuntu@$IP" 'tail -1 ~/haotian_research-1/logs/flowgrpo_runner.log 2>/dev/null | tr -d "\r" | head -c 100; echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' 2>/dev/null || echo "ssh unreachable")
    log "heartbeat: $(echo "$HEART" | tr '\n' ' | ')"
done

# Pull logs and any checkpoints. Best-effort.
log "Rsyncing logs..."
rsync -az -e "ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no" \
    "ubuntu@$IP:~/haotian_research-1/logs/" ./logs/ 2>&1 | tee -a "$LOG" || log "WARN: log rsync failed"

log "Rsyncing checkpoints..."
rsync -az -e "ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no" \
    "ubuntu@$IP:~/haotian_research-1/outputs/checkpoints/flowgrpo/" ./outputs/checkpoints/flowgrpo/ 2>&1 | tee -a "$LOG" || log "WARN: ckpt rsync failed"

# Also pull the raw flow_grpo logs dir (has tensorboard events, etc.)
log "Rsyncing flow_grpo run logs..."
rsync -az -e "ssh -i $HOME/.ssh/lambda_claude -o StrictHostKeyChecking=no" \
    "ubuntu@$IP:~/haotian_research-1/repos/flow_grpo/logs/compressibility/" ./outputs/flowgrpo_runlogs/ 2>&1 | tee -a "$LOG" || log "WARN: runlogs rsync failed"

# Terminate.
log "Terminating Lambda instance $INST..."
RESP=$(curl -s -u "$LAMBDA_API_KEY:" \
    -H "Content-Type: application/json" \
    -d "{\"instance_ids\":[\"$INST\"]}" \
    https://cloud.lambdalabs.com/api/v1/instance-operations/terminate)
echo "$RESP" | tee -a "$LOG"
rm -f .lambda_instance .lambda_ip

log "=== babysit done ==="
