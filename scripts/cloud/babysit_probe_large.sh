#!/bin/bash
# One-shot probe: does FlowGRPO group=8 fit SD3.5-LARGE on an 80GB box?
# Launches a box, runs run_flowgrpo_clip.sh for 1h with FG_MODEL=large,
# reports checkpoint count + peak memory + any OOM/auth errors, terminates.
# Usage: bash scripts/cloud/babysit_probe_large.sh

set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
: "${LAMBDA_API_KEY:?}"; : "${HF_TOKEN:?}"

KEY_NAME="claude-lambda-20260419-200846"
SSH_KEY="$HOME/.ssh/lambda_claude"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"
INST_FILE=".lambda_instance_probe"; IP_FILE=".lambda_ip_probe"
mkdir -p logs outputs
LOG="logs/babysit_probe_large.log"
log() { echo "[$(date +%H:%M:%S)][probe] $*" | tee -a "$LOG"; }

terminate_box() {
    [ -f "$INST_FILE" ] || return 0
    local id st a; id="$(cat "$INST_FILE")"
    for a in 1 2 3 4 5; do
        log "terminating $id (attempt $a)"
        curl -s --max-time 30 -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" \
            -d "{\"instance_ids\":[\"$id\"]}" \
            https://cloud.lambdalabs.com/api/v1/instance-operations/terminate >> "$LOG" 2>&1
        sleep 15
        st=$(curl -s --max-time 30 -u "$LAMBDA_API_KEY:" \
            "https://cloud.lambdalabs.com/api/v1/instances/$id" \
            | python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('status','QUERYFAIL'))" 2>/dev/null || echo QUERYFAIL)
        log "  post-terminate status: '$st'"
        case "$st" in terminating|terminated) rm -f "$INST_FILE" "$IP_FILE"; return 0;; esac
    done
    log "ERROR: unconfirmed termination of $id"
}

log "=== large probe start ==="
if [ ! -f "$INST_FILE" ]; then
    CANDIDATES=$(curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instance-types | python3 -c "
import json, sys
d = json.load(sys.stdin)['data']
for t in ['gpu_1x_gh200']:
    if t in d:
        for r in d[t]['regions_with_capacity_available']:
            print(t, r['name'])
")
    [ -n "$CANDIDATES" ] || { log "FATAL: no capacity"; exit 1; }
    LAUNCHED=0
    while read -r ITYPE IREGION; do
        log "trying $ITYPE @ $IREGION ..."
        RESP=$(curl -s -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" -X POST \
            -d "{\"region_name\":\"$IREGION\",\"instance_type_name\":\"$ITYPE\",\"ssh_key_names\":[\"$KEY_NAME\"],\"quantity\":1}" \
            https://cloud.lambdalabs.com/api/v1/instance-operations/launch)
        ID=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('instance_ids',[''])[0] or '')")
        if [ -n "$ID" ]; then echo "$ID" > "$INST_FILE"; log "launched $ID ($ITYPE @ $IREGION)"; LAUNCHED=1; break
        else log "  refused: $(echo "$RESP" | head -c 140)"; fi
    done <<< "$CANDIDATES"
    [ "$LAUNCHED" = 1 ] || { log "FATAL: all refused"; exit 1; }
fi
INST="$(cat "$INST_FILE")"
if [ ! -f "$IP_FILE" ]; then
    for i in $(seq 1 90); do
        sleep 10
        INFO=$(curl -s -u "$LAMBDA_API_KEY:" "https://cloud.lambdalabs.com/api/v1/instances/$INST")
        STATUS=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('status','?'))")
        IP=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('ip','') or '')")
        if [ "$STATUS" = "active" ] && [ -n "$IP" ]; then echo "$IP" > "$IP_FILE"; break; fi
    done
    [ -f "$IP_FILE" ] || { log "FATAL: never active"; exit 1; }
fi
IP="$(cat "$IP_FILE")"
log "instance $INST @ $IP"

for i in $(seq 1 40); do
    if $SSH "ubuntu@$IP" 'echo ok' >/dev/null 2>&1; then log "SSH up."; break; fi
    sleep 15
done

RUNNER_LOG="logs/probe_large_runner.log"
STATE=$($SSH "ubuntu@$IP" "pgrep -f 'run_flowgrpo_clip[.]sh' >/dev/null && echo RUNNING || echo IDLE" 2>/dev/null || echo IDLE)
if [ "$STATE" = "IDLE" ] && ! $SSH "ubuntu@$IP" "grep -q 'FlowGRPO+CLIP run done' ~/haotian_research-1/$RUNNER_LOG 2>/dev/null"; then
    $SSH "ubuntu@$IP" 'mkdir -p ~/haotian_research-1/logs ~/haotian_research-1/outputs'
    rsync -az -e "$SSH" --delete scripts/ "ubuntu@$IP:/home/ubuntu/haotian_research-1/scripts/" >> "$LOG" 2>&1
    $SSH "ubuntu@$IP" "cd ~/haotian_research-1 && \
        setsid nohup env HF_TOKEN='$HF_TOKEN' FG_GROUP=8 FG_NBPE=8 FG_SEED=42 \
        FG_MODEL='stabilityai/stable-diffusion-3.5-large' \
        bash scripts/cloud/run_flowgrpo_clip.sh 1 \
        > $RUNNER_LOG 2>&1 < /dev/null & echo kicked" \
        || { log "FATAL: kick failed"; terminate_box; exit 1; }
    sleep 20
    # Only a SUCCESSFUL ssh round-trip (sentinel echoed) may declare the runner
    # dead; a transient ssh failure must never terminate the box.
    VERDICT=""
    for i in 1 2 3 4 5; do
        OUT=$($SSH "ubuntu@$IP" "pgrep -f 'run_flowgrpo_clip[.]sh' | head -1; echo __SSHOK__" 2>/dev/null || true)
        if echo "$OUT" | grep -q __SSHOK__; then
            PID=$(echo "$OUT" | grep -v __SSHOK__ | head -1)
            if [ -n "$PID" ]; then VERDICT="alive"; else VERDICT="dead"; fi
            break
        fi
        sleep 12
    done
    if [ "$VERDICT" = "dead" ]; then
        log "runner exited early — capturing runner log, falling through to pull:"
        $SSH "ubuntu@$IP" "tail -40 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null" 2>/dev/null | tee -a "$LOG"
    elif [ -z "$VERDICT" ]; then
        log "WARN: ssh flaky, runner state unknown — proceeding to watch loop"
    else
        log "probe runner alive"
    fi
fi

# watch up to 2.5h (setup + bigger download + 1h train + ckpt copy)
DEADLINE=$(( $(date +%s) + 150 * 60 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 180
    DONE=$($SSH "ubuntu@$IP" "grep -c 'FlowGRPO+CLIP run done' ~/haotian_research-1/$RUNNER_LOG 2>/dev/null" 2>/dev/null || echo 0)
    [ "$DONE" != "0" ] && { log "probe finished"; break; }
    HB=$($SSH "ubuntu@$IP" "nvidia-smi --query-gpu=memory.used --format=csv,noheader" 2>/dev/null || echo unreachable)
    log "hb mem: $HB"
done

log "pulling runner log (with retries)..."
for a in 1 2 3 4; do
    rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/$RUNNER_LOG" ./logs/ >> "$LOG" 2>&1 && break
    log "  runner-log pull retry $a"; sleep 10
done
CKPTS=$($SSH "ubuntu@$IP" "ls ~/haotian_research-1/outputs/checkpoints/flowgrpo_clip 2>/dev/null | grep -c '^checkpoint-'" 2>/dev/null || echo "?")
log "verdict: checkpoints=$CKPTS oom_lines=$(grep -ciE 'out of memory' logs/probe_large_runner.log 2>/dev/null || echo 0) auth_lines=$(grep -ciE '401|gated|access to model' logs/probe_large_runner.log 2>/dev/null || echo 0)"
log "last error context:"
grep -B2 -A12 "Error\|error\|Traceback" logs/probe_large_runner.log 2>/dev/null | grep -v "^--$" | tail -30 | tee -a "$LOG" || true
terminate_box
log "=== large probe done ==="