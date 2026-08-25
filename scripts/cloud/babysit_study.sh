#!/bin/bash
# End-to-end driver for ONE reward of the TFG-vs-GRPO study. Parallel-safe:
# per-reward state files (.lambda_instance_<key>), so three of these can run
# concurrently on three boxes.
#
#   launch 80GB box (capacity-aware across regions/types) -> rsync scripts ->
#   kick run_study_reward.sh -> watch -> pull matrix JSON + endpoint LoRAs +
#   sample images -> terminate.
#
# Usage: bash scripts/cloud/babysit_study.sh <clip|clip_ocr|pickscore> [WALL_HOURS]

set -uo pipefail
cd "$(dirname "$0")/../.."

set -a; source .env; set +a
: "${LAMBDA_API_KEY:?LAMBDA_API_KEY missing}"
: "${HF_TOKEN:?HF_TOKEN missing}"

KEY_NAME="claude-lambda-20260419-200846"
SSH_KEY="$HOME/.ssh/lambda_claude"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"

REWARD_KEY="${1:?usage: babysit_study.sh <clip|clip_ocr|pickscore> [WALL_HOURS]}"
WALL_HOURS="${2:-4}"
export FG_GROUP=8 FG_NBPE=8

INST_FILE=".lambda_instance_$REWARD_KEY"
IP_FILE=".lambda_ip_$REWARD_KEY"
mkdir -p logs outputs
LOG="logs/babysit_study_${REWARD_KEY}.log"
log() { echo "[$(date +%H:%M:%S)][$REWARD_KEY] $*" | tee -a "$LOG"; }

terminate_box() {
    [ -f "$INST_FILE" ] || return 0
    local id st a
    id="$(cat "$INST_FILE")"
    # Verify + retry: a single silently-failed curl here leaks a $3.29/hr box.
    for a in 1 2 3 4 5; do
        log "terminating $id (attempt $a)"
        curl -s --max-time 30 -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" \
            -d "{\"instance_ids\":[\"$id\"]}" \
            https://cloud.lambdalabs.com/api/v1/instance-operations/terminate >> "$LOG" 2>&1
        sleep 15
        st=$(curl -s --max-time 30 -u "$LAMBDA_API_KEY:" \
            "https://cloud.lambdalabs.com/api/v1/instances/$id" \
            | python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('status',''))" 2>/dev/null || echo "")
        log "  post-terminate status: '${st:-gone}'"
        case "$st" in terminating|terminated|"") rm -f "$INST_FILE" "$IP_FILE"; return 0;; esac
    done
    log "ERROR: could not confirm termination of $id — CHECK THE DASHBOARD"
}
trap 'rc=$?; if [ $rc -ne 0 ]; then log "EXIT rc=$rc — cleanup"; terminate_box; fi' EXIT

log "=== study driver start: reward=$REWARD_KEY group=8 ${WALL_HOURS}h train ==="

# --- 1. Launch: pick an 80GB type+region with live capacity ---
if [ ! -f "$INST_FILE" ]; then
    CANDIDATES=$(curl -s -u "$LAMBDA_API_KEY:" \
        https://cloud.lambdalabs.com/api/v1/instance-types | python3 -c "
import json, sys
d = json.load(sys.stdin)['data']
prefer = ['gpu_1x_h100_pcie', 'gpu_1x_h100_sxm5', 'gpu_1x_gh200']
for t in prefer:
    if t in d:
        for r in d[t]['regions_with_capacity_available']:
            print(t, r['name'])
" )
    [ -n "$CANDIDATES" ] || { log "FATAL: no 80GB capacity anywhere"; exit 1; }
    LAUNCHED=0
    while read -r ITYPE IREGION; do
        log "trying $ITYPE @ $IREGION ..."
        RESP=$(curl -s -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" -X POST \
            -d "{\"region_name\":\"$IREGION\",\"instance_type_name\":\"$ITYPE\",\"ssh_key_names\":[\"$KEY_NAME\"],\"quantity\":1}" \
            https://cloud.lambdalabs.com/api/v1/instance-operations/launch)
        ID=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('instance_ids',[''])[0] or '')")
        if [ -n "$ID" ]; then
            echo "$ID" > "$INST_FILE"
            log "launched $ID ($ITYPE @ $IREGION)"
            LAUNCHED=1; break
        else
            log "  launch refused: $(echo "$RESP" | head -c 200)"
        fi
    done <<< "$CANDIDATES"
    [ "$LAUNCHED" = 1 ] || { log "FATAL: all launch attempts refused"; exit 1; }

    INST="$(cat "$INST_FILE")"
    for i in $(seq 1 90); do
        sleep 10
        INFO=$(curl -s -u "$LAMBDA_API_KEY:" "https://cloud.lambdalabs.com/api/v1/instances/$INST")
        STATUS=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('status','?'))")
        IP=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('ip','') or '')")
        [ $((i % 6)) = 0 ] && log "  boot: status=$STATUS ip=${IP:-pending}"
        if [ "$STATUS" = "active" ] && [ -n "$IP" ]; then echo "$IP" > "$IP_FILE"; break; fi
    done
    [ -f "$IP_FILE" ] || { log "FATAL: never went active"; exit 1; }
fi
IP="$(cat "$IP_FILE")"; INST="$(cat "$INST_FILE")"
log "instance $INST @ $IP"

# --- 2. SSH up ---
SSH_OK=0
for i in $(seq 1 40); do
    if $SSH "ubuntu@$IP" 'echo ok' >/dev/null 2>&1; then SSH_OK=1; log "SSH up."; break; fi
    sleep 15
done
[ "$SSH_OK" = 1 ] || { log "FATAL: SSH never came up"; exit 1; }

# --- 3. Push scripts ---
$SSH "ubuntu@$IP" 'mkdir -p ~/haotian_research-1/logs ~/haotian_research-1/outputs'
rsync -az -e "$SSH" --delete scripts/ "ubuntu@$IP:/home/ubuntu/haotian_research-1/scripts/" >> "$LOG" 2>&1

# --- 4. Kick orchestrator ---
RUNNER_LOG="logs/study_${REWARD_KEY}_runner.log"
DONE_SENTINEL="=== STUDY[$REWARD_KEY] done ==="
log "kicking run_study_reward.sh $REWARD_KEY $WALL_HOURS ..."
# Detach via kick_study.sh ON the box — a `cd && nohup ... &` one-liner
# backgrounds the whole compound, and the waiting subshell holds the ssh
# session's pipes open until TCP timeout (~35 min) -> "kick failed".
# kick_study.sh binds the redirections to the runner itself and setsids it,
# so this ssh returns in <1s (verified locally via command substitution).
$SSH "ubuntu@$IP" "HF_TOKEN='$HF_TOKEN' FG_GROUP=8 FG_NBPE=8 \
    bash ~/haotian_research-1/scripts/cloud/kick_study.sh $REWARD_KEY $WALL_HOURS" \
    || { log "FATAL: kick failed"; exit 1; }
sleep 20
ALIVE=$($SSH "ubuntu@$IP" "pgrep -f 'run_study_reward.sh $REWARD_KEY' | head -1" 2>/dev/null || true)
[ -n "$ALIVE" ] || { log "FATAL: runner not alive after kick"; exit 1; }
log "runner alive (pid $ALIVE)"

# --- 5. Watch (train wall + eval ~3h + setup ~1h + slack) ---
DEADLINE=$(( $(date +%s) + (WALL_HOURS + 6) * 3600 ))
FINISHED=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 180
    TAIL=$($SSH "ubuntu@$IP" "tail -8 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null" 2>/dev/null || true)
    if echo "$TAIL" | grep -qF "$DONE_SENTINEL"; then log "orchestrator finished."; FINISHED=1; break; fi
    if echo "$TAIL" | grep -qF "FATAL:"; then log "orchestrator FATAL — pulling what exists"; break; fi
    HB=$($SSH "ubuntu@$IP" "tail -1 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null | tr -d '\r' | head -c 110; echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" 2>/dev/null || echo "ssh unreachable")
    log "hb: $(echo "$HB" | tr '\n' '|')"
done
[ "$FINISHED" = 1 ] || log "WARN: exited watch loop without DONE sentinel"

# --- 6. Pull: matrix JSON, runner logs, endpoint LoRAs, sample images ---
log "pulling results..."
rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/study_${REWARD_KEY}_matrix.json" ./outputs/ >> "$LOG" 2>&1 \
    || log "WARN: matrix JSON pull failed"
rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/logs/" ./logs/ >> "$LOG" 2>&1 || true

# Endpoint LoRAs only (full checkpoint tree is ~15GB; endpoints are ~100MB).
ENDPOINTS=$(python3 -c "
import json
try:
    d = json.load(open('outputs/study_${REWARD_KEY}_matrix.json'))
    print(' '.join(d['meta'].get('rl_endpoints', [])))
except Exception:
    pass" 2>/dev/null)
case "$REWARD_KEY" in
  clip)      CKPT_SUB=flowgrpo_clip ;;
  clip_ocr)  CKPT_SUB=flowgrpo_clip_ocr ;;
  pickscore) CKPT_SUB=flowgrpo_pickscore ;;
esac
mkdir -p "outputs/checkpoints/${CKPT_SUB}_g8"
for ep in $ENDPOINTS; do
    log "pulling LoRA $ep ..."
    rsync -az -e "$SSH" \
        "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/checkpoints/$CKPT_SUB/$ep" \
        "outputs/checkpoints/${CKPT_SUB}_g8/" >> "$LOG" 2>&1 || log "WARN: LoRA $ep pull failed"
done

# Sample images: seed base 0, first 6 prompts of every arm (~25MB).
log "pulling sample images..."
rsync -az -e "$SSH" \
    --include='*/' --include='/*/0/00[0-5].png' --exclude='*' \
    "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/study_${REWARD_KEY}_images/" \
    "outputs/study_${REWARD_KEY}_images/" >> "$LOG" 2>&1 || log "WARN: image pull failed"

# --- 7. Terminate ---
terminate_box
trap - EXIT
log "=== study driver done: $REWARD_KEY ==="
if [ -f "outputs/study_${REWARD_KEY}_matrix.json" ]; then
    python3 - <<PY
import json
d = json.load(open("outputs/study_${REWARD_KEY}_matrix.json"))
s = d.get("summary")
if s:
    prim = d["meta"]["reward"]
    print("RESULT [$REWARD_KEY] primary=" + prim)
    for arm, e in s.items():
        m = e[prim]
        extra = ""
        if "delta_vs_no_lora" in m:
            extra = f"  d={m['delta_vs_no_lora']:+.4f} z={m['z']:+.1f} wins={m['wins']}/{m['n']}"
        print(f"  {arm:32s} mean={m['mean']:+.4f}{extra}")
else:
    print("RESULT [$REWARD_KEY]: no summary block (eval incomplete)")
PY
fi
