#!/bin/bash
# Driver for one role of the reward-ensemble experiment. Resumable.
# Train roles pull the final LoRA home; the eval role uploads all three LoRAs.
# Usage: bash scripts/cloud/babysit_ens.sh <t42|t1042|t2042|eval>

set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
: "${LAMBDA_API_KEY:?}"; : "${HF_TOKEN:?}"

ROLE="${1:?usage: babysit_ens.sh <t42|t1042|t2042|eval>}"
KEY_NAME="claude-lambda-20260419-200846"
SSH_KEY="$HOME/.ssh/lambda_claude"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"
INST_FILE=".lambda_instance_ens_$ROLE"; IP_FILE=".lambda_ip_ens_$ROLE"
mkdir -p logs outputs
LOG="logs/babysit_ens_$ROLE.log"
log() { echo "[$(date +%H:%M:%S)][ens-$ROLE] $*" | tee -a "$LOG"; }

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

log "=== ens driver start (role $ROLE) ==="
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

RUNNER_LOG="logs/ens_${ROLE}_runner.log"
DONE_SENTINEL="=== ENS[$ROLE] done ==="
STATE=$($SSH "ubuntu@$IP" "grep -qF '$DONE_SENTINEL' ~/haotian_research-1/$RUNNER_LOG 2>/dev/null && echo DONE || (pgrep -f 'run_ens[.]sh $ROLE' >/dev/null && echo RUNNING || echo IDLE)" 2>/dev/null || echo IDLE)
log "remote state: $STATE"

if [ "$STATE" = "IDLE" ]; then
    $SSH "ubuntu@$IP" 'mkdir -p ~/haotian_research-1/logs ~/haotian_research-1/outputs/checkpoints'
    rsync -az -e "$SSH" --delete scripts/ "ubuntu@$IP:/home/ubuntu/haotian_research-1/scripts/" >> "$LOG" 2>&1
    if [ "$ROLE" = "eval" ]; then
        log "uploading 3 ensemble LoRAs..."
        for SEED in 42 1042 2042; do
            SRC="outputs/checkpoints/flowgrpo_ensemble_s${SEED}"
            [ -d "$SRC" ] || { log "FATAL: $SRC missing locally (train roles not pulled yet?)"; exit 1; }
            rsync -az --partial -e "$SSH" "$SRC" \
                "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/checkpoints/" >> "$LOG" 2>&1 \
                || { log "FATAL: LoRA s$SEED upload failed"; exit 1; }
        done
    fi
    $SSH "ubuntu@$IP" "cd ~/haotian_research-1 && mkdir -p logs && \
        setsid nohup env HF_TOKEN='$HF_TOKEN' FG_GROUP=8 FG_NBPE=8 \
        bash scripts/cloud/run_ens.sh $ROLE \
        > $RUNNER_LOG 2>&1 < /dev/null & echo kicked" \
        || { log "FATAL: kick failed"; exit 1; }
    sleep 20
    ALIVE=$($SSH "ubuntu@$IP" "pgrep -f 'run_ens[.]sh $ROLE' | head -1" 2>/dev/null || true)
    [ -n "$ALIVE" ] || { log "FATAL: runner not alive"; exit 1; }
    log "runner alive (pid $ALIVE)"
fi

if [ "$STATE" != "DONE" ]; then
    DEADLINE=$(( $(date +%s) + 9 * 3600 ))
    while [ "$(date +%s)" -lt "$DEADLINE" ]; do
        sleep 240
        TAIL=$($SSH "ubuntu@$IP" "tail -6 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null" 2>/dev/null || true)
        if echo "$TAIL" | grep -qF "$DONE_SENTINEL"; then log "runner finished."; break; fi
        if echo "$TAIL" | grep -qF "FATAL:"; then log "runner FATAL"; break; fi
        HB=$($SSH "ubuntu@$IP" "tail -1 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null | tr -d '\r' | head -c 90" 2>/dev/null || echo unreachable)
        log "hb: $HB"
    done
fi

log "pulling..."
OK=1
if [ "$ROLE" = "eval" ]; then
    for a in 1 2 3; do
        rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/study_ensemble_matrix.json" ./outputs/ >> "$LOG" 2>&1 \
            && { log "  matrix ok"; break; }
        sleep 8
    done
    [ -f outputs/study_ensemble_matrix.json ] || OK=0
    rsync -az --partial -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/study_ensemble_images.tar.gz" ./outputs/ >> "$LOG" 2>&1 \
        && log "  image tar ok" || log "  WARN: image tar pull failed"
else
    SEED="${ROLE#t}"
    FINAL=$($SSH "ubuntu@$IP" "ls ~/haotian_research-1/outputs/checkpoints/flowgrpo_ensemble_s${SEED} 2>/dev/null | grep '^checkpoint-' | sort -t- -k2 -n | tail -1" 2>/dev/null || true)
    if [ -n "$FINAL" ]; then
        mkdir -p "outputs/checkpoints/flowgrpo_ensemble_s${SEED}"
        rsync -az --partial -e "$SSH" \
            "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/checkpoints/flowgrpo_ensemble_s${SEED}/$FINAL" \
            "outputs/checkpoints/flowgrpo_ensemble_s${SEED}/" >> "$LOG" 2>&1 \
            && log "  LoRA s$SEED/$FINAL ok" || OK=0
    else
        OK=0
    fi
fi
rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/$RUNNER_LOG" ./logs/ >> "$LOG" 2>&1 || true

if [ "$OK" = 1 ]; then terminate_box; else log "NOT terminating: expected artifacts missing"; fi
log "=== ens driver done ($ROLE) ==="
