#!/bin/bash
# Driver for the detector-eval pass. Resumable: rerun to reattach (skips
# launch/upload/kick when already done). Terminates only after the detector
# JSONs are confirmed local.
# Usage: bash scripts/cloud/babysit_detector.sh

set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
: "${LAMBDA_API_KEY:?}"; : "${HF_TOKEN:?}"

KEY_NAME="claude-lambda-20260419-200846"
SSH_KEY="$HOME/.ssh/lambda_claude"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"
INST_FILE=".lambda_instance_det"; IP_FILE=".lambda_ip_det"
mkdir -p logs outputs
LOG="logs/babysit_detector.log"
log() { echo "[$(date +%H:%M:%S)][det] $*" | tee -a "$LOG"; }

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

log "=== detector driver start ==="
if [ ! -f "$INST_FILE" ]; then
    CANDIDATES=$(curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instance-types | python3 -c "
import json, sys
d = json.load(sys.stdin)['data']
for t in ['gpu_1x_a100_sxm4', 'gpu_1x_a100', 'gpu_1x_h100_pcie', 'gpu_1x_gh200', 'gpu_1x_h100_sxm5']:
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

SSH_OK=0
for i in $(seq 1 40); do
    if $SSH "ubuntu@$IP" 'echo ok' >/dev/null 2>&1; then SSH_OK=1; log "SSH up."; break; fi
    sleep 15
done
[ "$SSH_OK" = 1 ] || { log "FATAL: no SSH"; exit 1; }

RUNNER_LOG="logs/deteval_runner.log"
DONE_SENTINEL="=== DETEVAL done ==="
STATE=$($SSH "ubuntu@$IP" "grep -qF '$DONE_SENTINEL' ~/haotian_research-1/$RUNNER_LOG 2>/dev/null && echo DONE || (pgrep -f 'run_detector_eval[.]sh' >/dev/null && echo RUNNING || echo IDLE)" 2>/dev/null || echo IDLE)
log "remote state: $STATE"

if [ "$STATE" = "IDLE" ]; then
    log "uploading scripts + LoRAs + pickscore image tar (~1.7GB)..."
    $SSH "ubuntu@$IP" 'mkdir -p ~/haotian_research-1/logs ~/haotian_research-1/outputs ~/haotian_research-1/outputs/checkpoints'
    rsync -az -e "$SSH" --delete scripts/ "ubuntu@$IP:/home/ubuntu/haotian_research-1/scripts/" >> "$LOG" 2>&1
    rsync -az --partial -e "$SSH" outputs/checkpoints/flowgrpo_clip_pub_s42 outputs/checkpoints/flowgrpo_clip_pub_s2042 \
        "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/checkpoints/" >> "$LOG" 2>&1 \
        || { log "FATAL: LoRA upload failed"; exit 1; }
    rsync -az --partial -e "$SSH" outputs/study_pickscore_pub_images.tar.gz \
        "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/" >> "$LOG" 2>&1 \
        || { log "FATAL: tar upload failed"; exit 1; }
    log "kicking (detached)..."
    $SSH "ubuntu@$IP" "cd ~/haotian_research-1 && mkdir -p logs && \
        setsid nohup env HF_TOKEN='$HF_TOKEN' bash scripts/cloud/run_detector_eval.sh \
        > $RUNNER_LOG 2>&1 < /dev/null & echo kicked" \
        || { log "FATAL: kick failed"; exit 1; }
    sleep 20
    ALIVE=$($SSH "ubuntu@$IP" "pgrep -f 'run_detector_eval[.]sh' | head -1" 2>/dev/null || true)
    [ -n "$ALIVE" ] || { log "FATAL: runner not alive"; exit 1; }
    log "runner alive (pid $ALIVE)"
fi

if [ "$STATE" != "DONE" ]; then
    DEADLINE=$(( $(date +%s) + 11 * 3600 ))
    FINISHED=0
    while [ "$(date +%s)" -lt "$DEADLINE" ]; do
        sleep 240
        TAIL=$($SSH "ubuntu@$IP" "tail -6 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null" 2>/dev/null || true)
        if echo "$TAIL" | grep -qF "$DONE_SENTINEL"; then log "runner finished."; FINISHED=1; break; fi
        if echo "$TAIL" | grep -qF "FATAL:"; then log "runner FATAL"; break; fi
        HB=$($SSH "ubuntu@$IP" "tail -1 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null | tr -d '\r' | head -c 90" 2>/dev/null || echo unreachable)
        log "hb: $HB"
    done
fi

log "pulling..."
for f in study_clip_pub_detector.json study_pickscore_pub_detector.json \
         study_clip_pub_external.json study_clip_pub_matrix_regen.json; do
    for a in 1 2 3; do
        rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/$f" ./outputs/ >> "$LOG" 2>&1 \
            && { log "  $f ok"; break; }
        sleep 8
    done
done
rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/$RUNNER_LOG" ./logs/ >> "$LOG" 2>&1 || true

if [ -f "outputs/study_clip_pub_detector.json" ] && [ -f "outputs/study_pickscore_pub_detector.json" ]; then
    terminate_box
else
    log "NOT terminating: detector JSONs missing locally"
fi
log "=== detector driver done ==="
