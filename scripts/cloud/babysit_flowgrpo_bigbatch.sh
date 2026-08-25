#!/bin/bash
# End-to-end driver for the big-batch (FG_GROUP=8) FlowGRPO experiment.
# Launches an 80GB H100, pushes scripts, kicks the sequential PickScore->CLIP
# orchestrator, watches it, pulls both eval JSONs, then terminates the box.
#
# Implements Haotian's suggestion: gradient accumulation + a much larger
# rollout/group size (group 2 -> 8) at matched wall-clock vs the failing runs.
#
# Usage: bash scripts/cloud/babysit_flowgrpo_bigbatch.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

set -a; source .env; set +a
: "${LAMBDA_API_KEY:?LAMBDA_API_KEY missing}"
: "${HF_TOKEN:?HF_TOKEN missing}"

INSTANCE_TYPE="gpu_1x_h100_pcie"   # 80GB; required for group=8
REGION="us-west-3"                 # had capacity at launch time
SSH_KEY_NAME="claude-lambda-20260419-200846"
WALL_HOURS_PER_REWARD=4            # matches the prior group=2 runs (controlled)
export FG_GROUP=8 FG_NBPE=8

KEY="$HOME/.ssh/lambda_claude"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"

mkdir -p logs outputs
LOG=logs/babysit_bigbatch.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Safety net: if we bail after the box is up but before the normal terminate,
# kill it so we never leak a running GPU. Cleared once we terminate normally.
terminate_box() {
    [ -f .lambda_instance ] || return 0
    local id; id="$(cat .lambda_instance)"
    log "cleanup: terminating $id"
    curl -s -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" \
        -d "{\"instance_ids\":[\"$id\"]}" \
        https://cloud.lambdalabs.com/api/v1/instance-operations/terminate | tee -a "$LOG"
    rm -f .lambda_instance .lambda_ip
}
trap 'rc=$?; if [ $rc -ne 0 ]; then log "EXIT rc=$rc — running cleanup"; terminate_box; fi' EXIT

log "=== big-batch driver start: $INSTANCE_TYPE @ $REGION, group=$FG_GROUP, ${WALL_HOURS_PER_REWARD}h/reward ==="

# --- 1. Launch (writes .lambda_ip / .lambda_instance) ---
if [ -f .lambda_instance ]; then
    log "Reusing existing instance $(cat .lambda_instance) @ $(cat .lambda_ip 2>/dev/null)"
else
    log "Launching..."
    bash scripts/cloud/launch_lambda.sh "$INSTANCE_TYPE" "$REGION" "$SSH_KEY_NAME" 2>&1 | tee -a "$LOG"
fi
[ -f .lambda_ip ] || { log "FATAL: launch failed, no .lambda_ip"; exit 1; }
IP="$(cat .lambda_ip)"; INST="$(cat .lambda_instance)"
log "Instance $INST @ $IP"

# --- 2. Wait for SSH ---
for i in $(seq 1 40); do
    if $SSH "ubuntu@$IP" 'echo ok' >/dev/null 2>&1; then log "SSH up."; break; fi
    sleep 15
done

# --- 3. Push scripts (flow_grpo is cloned on the box; we only need scripts/) ---
log "Rsyncing scripts/ to box..."
$SSH "ubuntu@$IP" 'mkdir -p ~/haotian_research-1/logs ~/haotian_research-1/outputs'
rsync -az -e "$SSH" --delete scripts/ "ubuntu@$IP:/home/ubuntu/haotian_research-1/scripts/" 2>&1 | tee -a "$LOG"

# --- 4. Kick the orchestrator under nohup ---
RUNNER_LOG="logs/flowgrpo_bigbatch_runner.log"
DONE_SENTINEL="=== BIGBATCH orchestrator done ==="
log "Kicking orchestrator (group=$FG_GROUP, ${WALL_HOURS_PER_REWARD}h/reward)..."
$SSH "ubuntu@$IP" "cd ~/haotian_research-1 && \
    HF_TOKEN='$HF_TOKEN' FG_GROUP=$FG_GROUP FG_NBPE=$FG_NBPE \
    nohup bash scripts/cloud/run_flowgrpo_bigbatch_both.sh $WALL_HOURS_PER_REWARD \
    > $RUNNER_LOG 2>&1 &" || { log "FATAL: kick failed"; exit 1; }
log "Orchestrator running. Watching $RUNNER_LOG ..."

# --- 5. Watch (budget = 2*wall + setup/eval headroom) ---
DEADLINE=$(( $(date +%s) + (2*WALL_HOURS_PER_REWARD + 4)*3600 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 120
    TAIL=$($SSH "ubuntu@$IP" "tail -6 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null" 2>/dev/null || true)
    if echo "$TAIL" | grep -qF "$DONE_SENTINEL"; then log "Orchestrator finished."; break; fi
    HB=$($SSH "ubuntu@$IP" "tail -1 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null | tr -d '\r' | head -c 110; echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" 2>/dev/null || echo "ssh unreachable")
    log "hb: $(echo "$HB" | tr '\n' '|')"
done

# --- 6. Pull results (both eval JSONs + logs) ---
log "Pulling eval JSONs + logs..."
for f in flowgrpo_pickscore_eval.json flowgrpo_clip_eval.json; do
    rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/$f" ./outputs/ 2>&1 | tee -a "$LOG" || log "WARN: $f pull failed"
done
rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/logs/" ./logs/ 2>&1 | tee -a "$LOG" || true

# --- 7. Terminate ---
log "Terminating $INST..."
curl -s -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" \
    -d "{\"instance_ids\":[\"$INST\"]}" \
    https://cloud.lambdalabs.com/api/v1/instance-operations/terminate | tee -a "$LOG"
rm -f .lambda_instance .lambda_ip
log "=== big-batch driver done ==="
for f in outputs/flowgrpo_pickscore_eval.json outputs/flowgrpo_clip_eval.json; do
    [ -f "$f" ] && log "RESULT $f: $(python3 -c "import json;d=json.load(open('$f'));print(d.get('rewards',d))" 2>/dev/null | head -c 300)"
done
