#!/bin/bash
# Driver for ONE reward of the publication-grade study. RESUMABLE: re-running
# after a local kill reattaches to the existing box (skips launch if the
# instance state file exists, skips kick if the runner is alive or done).
#
# Usage: bash scripts/cloud/babysit_pub.sh <clip|clip_ocr|pickscore> [WALL_HOURS_PER_SEED]

set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
: "${LAMBDA_API_KEY:?}"; : "${HF_TOKEN:?}"

KEY_NAME="claude-lambda-20260419-200846"
SSH_KEY="$HOME/.ssh/lambda_claude"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"

REWARD_KEY="${1:?usage: babysit_pub.sh <clip|clip_ocr|pickscore> [WALL_HOURS_PER_SEED]}"
WALL_HOURS="${2:-4}"

INST_FILE=".lambda_instance_pub_$REWARD_KEY"
IP_FILE=".lambda_ip_pub_$REWARD_KEY"
mkdir -p logs outputs
LOG="logs/babysit_pub_${REWARD_KEY}.log"
log() { echo "[$(date +%H:%M:%S)][pub-$REWARD_KEY] $*" | tee -a "$LOG"; }

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
    log "ERROR: unconfirmed termination of $id — CHECK DASHBOARD"
}
# NOTE: no auto-terminate trap on failure — a killed/failed local driver must
# NOT take the box down mid-run; rerun this script to reattach instead.

log "=== pub driver start: $REWARD_KEY (3 seeds x ${WALL_HOURS}h + matrix) ==="

# --- 1. Launch or reuse ---
if [ ! -f "$INST_FILE" ]; then
    CANDIDATES=$(curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instance-types | python3 -c "
import json, sys
d = json.load(sys.stdin)['data']
for t in ['gpu_1x_h100_pcie', 'gpu_1x_h100_sxm5', 'gpu_1x_gh200']:
    if t in d:
        for r in d[t]['regions_with_capacity_available']:
            print(t, r['name'])
")
    [ -n "$CANDIDATES" ] || { log "FATAL: no 80GB capacity anywhere"; exit 1; }
    LAUNCHED=0
    while read -r ITYPE IREGION; do
        log "trying $ITYPE @ $IREGION ..."
        RESP=$(curl -s -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" -X POST \
            -d "{\"region_name\":\"$IREGION\",\"instance_type_name\":\"$ITYPE\",\"ssh_key_names\":[\"$KEY_NAME\"],\"quantity\":1}" \
            https://cloud.lambdalabs.com/api/v1/instance-operations/launch)
        ID=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('instance_ids',[''])[0] or '')")
        if [ -n "$ID" ]; then echo "$ID" > "$INST_FILE"; log "launched $ID ($ITYPE @ $IREGION)"; LAUNCHED=1; break
        else log "  refused: $(echo "$RESP" | head -c 160)"; fi
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

# --- 2. SSH ---
SSH_OK=0
for i in $(seq 1 40); do
    if $SSH "ubuntu@$IP" 'echo ok' >/dev/null 2>&1; then SSH_OK=1; log "SSH up."; break; fi
    sleep 15
done
[ "$SSH_OK" = 1 ] || { log "FATAL: no SSH"; exit 1; }

RUNNER_LOG="logs/pub_${REWARD_KEY}_runner.log"
DONE_SENTINEL="=== PUB[$REWARD_KEY] done ==="

# --- 3. Kick, unless resuming ---
# pgrep pattern uses [.] so it cannot match this probe's own command line.
STATE=$($SSH "ubuntu@$IP" "grep -qF '$DONE_SENTINEL' ~/haotian_research-1/$RUNNER_LOG 2>/dev/null && echo DONE || (pgrep -f 'run_pub_reward[.]sh $REWARD_KEY' >/dev/null && echo RUNNING || echo IDLE)" 2>/dev/null || echo IDLE)
log "remote state: $STATE"
if [ "$STATE" = "IDLE" ]; then
    $SSH "ubuntu@$IP" 'mkdir -p ~/haotian_research-1/logs ~/haotian_research-1/outputs'
    rsync -az -e "$SSH" --delete scripts/ "ubuntu@$IP:/home/ubuntu/haotian_research-1/scripts/" >> "$LOG" 2>&1
    $SSH "ubuntu@$IP" "HF_TOKEN='$HF_TOKEN' bash ~/haotian_research-1/scripts/cloud/kick_pub.sh $REWARD_KEY $WALL_HOURS" \
        || { log "FATAL: kick failed"; exit 1; }
    sleep 20
    ALIVE=$($SSH "ubuntu@$IP" "pgrep -f 'run_pub_reward[.]sh $REWARD_KEY' | head -1" 2>/dev/null || true)
    [ -n "$ALIVE" ] || { log "FATAL: runner not alive after kick"; exit 1; }
    log "runner alive (pid $ALIVE)"
fi

# --- 4. Watch ---
if [ "$STATE" != "DONE" ]; then
    DEADLINE=$(( $(date +%s) + (3 * WALL_HOURS + 14) * 3600 ))
    FINISHED=0
    while [ "$(date +%s)" -lt "$DEADLINE" ]; do
        sleep 300
        TAIL=$($SSH "ubuntu@$IP" "tail -8 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null" 2>/dev/null || true)
        if echo "$TAIL" | grep -qF "$DONE_SENTINEL"; then log "runner finished."; FINISHED=1; break; fi
        if echo "$TAIL" | grep -qF "FATAL:"; then log "runner FATAL — pulling partials"; break; fi
        HB=$($SSH "ubuntu@$IP" "tail -1 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null | tr -d '\r' | head -c 100; echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" 2>/dev/null || echo "ssh unreachable")
        log "hb: $(echo "$HB" | tr '\n' '|')"
    done
    [ "$FINISHED" = 1 ] || log "WARN: watch ended without DONE sentinel"
fi

# --- 5. Pull ---
log "pulling results..."
for f in "study_${REWARD_KEY}_pub_matrix.json" "study_${REWARD_KEY}_pub_external.json"; do
    rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/$f" ./outputs/ >> "$LOG" 2>&1 \
        && log "  $f ok" || log "  WARN: $f pull failed"
done
rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/$RUNNER_LOG" ./logs/ >> "$LOG" 2>&1 || true

case "$REWARD_KEY" in
  clip)      CKPT_SUB=flowgrpo_clip ;;
  clip_ocr)  CKPT_SUB=flowgrpo_clip_ocr ;;
  pickscore) CKPT_SUB=flowgrpo_pickscore ;;
esac
for SEED in 42 1042 2042; do
    FINAL=$($SSH "ubuntu@$IP" "ls ~/haotian_research-1/outputs/checkpoints/${CKPT_SUB}_s${SEED} 2>/dev/null | grep '^checkpoint-' | sort -t- -k2 -n | tail -1" 2>/dev/null || true)
    [ -n "$FINAL" ] || { log "WARN: no final ckpt for seed $SEED"; continue; }
    mkdir -p "outputs/checkpoints/${CKPT_SUB}_pub_s${SEED}"
    rsync -az -e "$SSH" \
        "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/checkpoints/${CKPT_SUB}_s${SEED}/$FINAL" \
        "outputs/checkpoints/${CKPT_SUB}_pub_s${SEED}/" >> "$LOG" 2>&1 \
        && log "  LoRA s$SEED/$FINAL ok" || log "  WARN: LoRA s$SEED pull failed"
done

log "pulling image tar (large, tolerant)..."
rsync -az --partial -e "$SSH" \
    "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/study_${REWARD_KEY}_pub_images.tar.gz" \
    ./outputs/ >> "$LOG" 2>&1 && log "  image tar ok" || log "  WARN: image tar pull failed"

# --- 6. Terminate only if the matrix JSON made it home ---
if [ -f "outputs/study_${REWARD_KEY}_pub_matrix.json" ]; then
    terminate_box
else
    log "NOT terminating: matrix JSON missing locally — box left up for manual recovery"
fi
log "=== pub driver done: $REWARD_KEY ==="
