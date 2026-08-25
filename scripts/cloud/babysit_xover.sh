#!/bin/bash
# Driver for one role of the budget-crossover experiment. Resumable.
# Usage: bash scripts/cloud/babysit_xover.sh <a|b>

set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
: "${LAMBDA_API_KEY:?}"; : "${HF_TOKEN:?}"

ROLE="${1:?usage: babysit_xover.sh <a|b>}"
KEY_NAME="claude-lambda-20260419-200846"
SSH_KEY="$HOME/.ssh/lambda_claude"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"
INST_FILE=".lambda_instance_xover_$ROLE"; IP_FILE=".lambda_ip_xover_$ROLE"
mkdir -p logs outputs
LOG="logs/babysit_xover_$ROLE.log"
log() { echo "[$(date +%H:%M:%S)][xover-$ROLE] $*" | tee -a "$LOG"; }

if [ "$ROLE" = "a" ]; then
    EXPECT_JSONS="study_xover8h_matrix.json"
    PULL_TAGS="8h_s42 8h_s1042"
    WATCH_HOURS=22
elif [ "$ROLE" = "c" ]; then
    EXPECT_JSONS="study_xover32h_matrix.json"
    PULL_TAGS="32h_s42"
    WATCH_HOURS=38
else
    EXPECT_JSONS="study_xover16h_matrix.json study_pickscore_rhoceil_matrix.json"
    PULL_TAGS="16h_s42"
    WATCH_HOURS=22
fi

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

log "=== xover driver start (role $ROLE) ==="
if [ ! -f "$INST_FILE" ]; then
    CANDIDATES=$(curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instance-types | python3 -c "
import json, sys
d = json.load(sys.stdin)['data']
for t in ['gpu_1x_h100_pcie', 'gpu_1x_h100_sxm5', 'gpu_1x_gh200']:
    if t in d:
        for r in d[t]['regions_with_capacity_available']:
            print(t, r['name'])
")
    [ -n "$CANDIDATES" ] || { log "FATAL: no 80GB capacity"; exit 1; }
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

SSH_OK=0
for i in $(seq 1 40); do
    if $SSH "ubuntu@$IP" 'echo ok' >/dev/null 2>&1; then SSH_OK=1; log "SSH up."; break; fi
    sleep 15
done
[ "$SSH_OK" = 1 ] || { log "FATAL: no SSH"; exit 1; }

RUNNER_LOG="logs/xover_${ROLE}_runner.log"
DONE_SENTINEL="=== XOVER[$ROLE] done ==="
STATE=$($SSH "ubuntu@$IP" "grep -qF '$DONE_SENTINEL' ~/haotian_research-1/$RUNNER_LOG 2>/dev/null && echo DONE || (pgrep -f 'run_xover[.]sh $ROLE' >/dev/null && echo RUNNING || echo IDLE)" 2>/dev/null || echo IDLE)
log "remote state: $STATE"

if [ "$STATE" = "IDLE" ]; then
    $SSH "ubuntu@$IP" 'mkdir -p ~/haotian_research-1/logs ~/haotian_research-1/outputs'
    rsync -az -e "$SSH" --delete scripts/ "ubuntu@$IP:/home/ubuntu/haotian_research-1/scripts/" >> "$LOG" 2>&1
    $SSH "ubuntu@$IP" "cd ~/haotian_research-1 && mkdir -p logs && \
        setsid nohup env HF_TOKEN='$HF_TOKEN' FG_GROUP=8 FG_NBPE=8 \
        bash scripts/cloud/run_xover.sh $ROLE \
        > $RUNNER_LOG 2>&1 < /dev/null & echo kicked" \
        || { log "FATAL: kick failed"; exit 1; }
    sleep 20
    ALIVE=$($SSH "ubuntu@$IP" "pgrep -f 'run_xover[.]sh $ROLE' | head -1" 2>/dev/null || true)
    [ -n "$ALIVE" ] || { log "FATAL: runner not alive"; exit 1; }
    log "runner alive (pid $ALIVE)"
fi

if [ "$STATE" != "DONE" ]; then
    DEADLINE=$(( $(date +%s) + WATCH_HOURS * 3600 ))
    while [ "$(date +%s)" -lt "$DEADLINE" ]; do
        sleep 300
        TAIL=$($SSH "ubuntu@$IP" "tail -6 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null" 2>/dev/null || true)
        if echo "$TAIL" | grep -qF "$DONE_SENTINEL"; then log "runner finished."; break; fi
        if echo "$TAIL" | grep -qF "FATAL:"; then log "runner FATAL"; break; fi
        HB=$($SSH "ubuntu@$IP" "tail -1 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null | tr -d '\r' | head -c 90" 2>/dev/null || echo unreachable)
        log "hb: $HB"
    done
fi

log "pulling..."
OK=1
for f in $EXPECT_JSONS; do
    for a in 1 2 3; do
        rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/$f" ./outputs/ >> "$LOG" 2>&1 \
            && { log "  $f ok"; break; }
        sleep 8
    done
    [ -f "outputs/$f" ] || OK=0
done
rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/$RUNNER_LOG" ./logs/ >> "$LOG" 2>&1 || true
for TAG in $PULL_TAGS; do
    FINAL=$($SSH "ubuntu@$IP" "ls ~/haotian_research-1/outputs/checkpoints/flowgrpo_clip_${TAG} 2>/dev/null | grep '^checkpoint-' | sort -t- -k2 -n | tail -1" 2>/dev/null || true)
    [ -n "$FINAL" ] || continue
    mkdir -p "outputs/checkpoints/flowgrpo_clip_xover_${TAG}"
    rsync -az --partial -e "$SSH" \
        "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/checkpoints/flowgrpo_clip_${TAG}/$FINAL" \
        "outputs/checkpoints/flowgrpo_clip_xover_${TAG}/" >> "$LOG" 2>&1 \
        && log "  LoRA $TAG/$FINAL ok" || log "  WARN: LoRA $TAG pull failed"
done

if [ "$OK" = 1 ]; then terminate_box; else log "NOT terminating: expected JSONs missing"; fi
log "=== xover driver done ($ROLE) ==="
