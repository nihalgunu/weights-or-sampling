#!/bin/bash
# Driver for the eval-only rho-extension run (TFG-PickScore rho 40/80).
# Any 1x GPU with ~40GB works (eval only, text encoders offloaded) — prefer
# cheap A100 40GB, fall back to 80GB types.
# Usage: bash scripts/cloud/babysit_mueval.sh

set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
: "${LAMBDA_API_KEY:?}"; : "${HF_TOKEN:?}"

KEY_NAME="claude-lambda-20260419-200846"
SSH_KEY="$HOME/.ssh/lambda_claude"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"
INST_FILE=".lambda_instance_mueval"; IP_FILE=".lambda_ip_mueval"
mkdir -p logs outputs
LOG="logs/babysit_mueval.log"
log() { echo "[$(date +%H:%M:%S)][mueval] $*" | tee -a "$LOG"; }

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
trap 'rc=$?; if [ $rc -ne 0 ]; then log "EXIT rc=$rc — cleanup"; terminate_box; fi' EXIT

log "=== mueval driver start ==="
if [ ! -f "$INST_FILE" ]; then
    CANDIDATES=$(curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instance-types | python3 -c "
import json, sys
d = json.load(sys.stdin)['data']
for t in ['gpu_1x_h100_pcie', 'gpu_1x_a100_sxm4', 'gpu_1x_a100', 'gpu_1x_h100_sxm5', 'gpu_1x_gh200']:
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
        else log "  refused: $(echo "$RESP" | head -c 160)"; fi
    done <<< "$CANDIDATES"
    [ "$LAUNCHED" = 1 ] || { log "FATAL: all refused"; exit 1; }
    INST="$(cat "$INST_FILE")"
    for i in $(seq 1 90); do
        sleep 10
        INFO=$(curl -s -u "$LAMBDA_API_KEY:" "https://cloud.lambdalabs.com/api/v1/instances/$INST")
        STATUS=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('status','?'))")
        IP=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('ip','') or '')")
        if [ "$STATUS" = "active" ] && [ -n "$IP" ]; then echo "$IP" > "$IP_FILE"; break; fi
    done
    [ -f "$IP_FILE" ] || { log "FATAL: never active"; exit 1; }
fi
IP="$(cat "$IP_FILE")"; INST="$(cat "$INST_FILE")"
log "instance $INST @ $IP"

SSH_OK=0
for i in $(seq 1 40); do
    if $SSH "ubuntu@$IP" 'echo ok' >/dev/null 2>&1; then SSH_OK=1; log "SSH up."; break; fi
    sleep 15
done
[ "$SSH_OK" = 1 ] || { log "FATAL: no SSH"; exit 1; }

$SSH "ubuntu@$IP" 'mkdir -p ~/haotian_research-1/logs ~/haotian_research-1/outputs'
rsync -az -e "$SSH" --delete scripts/ "ubuntu@$IP:/home/ubuntu/haotian_research-1/scripts/" >> "$LOG" 2>&1

# Detached kick (same mechanism as kick_study.sh — redirections bind to the
# whole simple command; parent shell exits immediately).
$SSH "ubuntu@$IP" "cd ~/haotian_research-1 && mkdir -p logs && \
    setsid nohup env HF_TOKEN='$HF_TOKEN' bash scripts/cloud/run_mu.sh \
    > logs/mueval_runner.log 2>&1 < /dev/null & echo kicked" \
    || { log "FATAL: kick failed"; exit 1; }
sleep 20
ALIVE=$($SSH "ubuntu@$IP" "pgrep -f run_mu.sh | head -1" 2>/dev/null || true)
[ -n "$ALIVE" ] || { log "FATAL: runner not alive"; exit 1; }
log "runner alive (pid $ALIVE)"

DEADLINE=$(( $(date +%s) + 8 * 3600 ))
FINISHED=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 120
    TAIL=$($SSH "ubuntu@$IP" "tail -6 ~/haotian_research-1/logs/mueval_runner.log 2>/dev/null" 2>/dev/null || true)
    if echo "$TAIL" | grep -qF "=== MUEVAL done ==="; then log "runner finished."; FINISHED=1; break; fi
    if echo "$TAIL" | grep -qE "FATAL|Traceback"; then log "runner ERROR:"; echo "$TAIL" | tee -a "$LOG"; break; fi
    HB=$($SSH "ubuntu@$IP" "tail -1 ~/haotian_research-1/logs/mueval_runner.log 2>/dev/null | tr -d '\r' | head -c 100" 2>/dev/null || echo unreachable)
    log "hb: $HB"
done

log "pulling..."
rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/study_mu_matrix.json" ./outputs/ >> "$LOG" 2>&1 || log "WARN: JSON pull failed"
rsync -az -e "$SSH" --include='*/' --include='/*/0/00[0-5].png' --exclude='*' \
    "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/study_mu_images_none/" \
    "outputs/study_mu_images_none/" >> "$LOG" 2>&1 || true
rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/logs/mueval_runner.log" ./logs/ >> "$LOG" 2>&1 || true

if [ -f "outputs/study_mu_matrix.json" ]; then terminate_box; else log "NOT terminating: JSON missing"; fi
trap - EXIT
log "=== mueval driver done ==="
