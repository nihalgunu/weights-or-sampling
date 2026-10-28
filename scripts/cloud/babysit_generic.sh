#!/bin/bash
# Generic resumable driver: launch 80GB box -> upload scripts (+ optional
# checkpoint dirs) -> detached kick of a runner -> watch -> pull expected
# outputs -> terminate only when they are all local.
#
# Usage:
#   bash scripts/cloud/babysit_generic.sh NAME RUNNER_REL SENTINEL \
#        "EXPECT_JSON1 [EXPECT_JSON2...]" "[UPLOAD_DIR1 UPLOAD_DIR2...]" [WATCH_HOURS]
# Env:
#   BABYSIT_SYNC="outputs/checkpoints/draft_clip_s42 outputs/study_x_matrix.json ..."
#        Project-relative paths mirrored box -> local on every poll (rsync, partial
#        files excluded) and pushed local -> box before a kick. With the trainers'
#        exact-state checkpoints (scripts/train_state.py) and the eval matrix's
#        per-arm resume, a lost box (preempted / unhealthy) costs at most one
#        checkpoint interval: the driver exits 3 ("instance lost"), kick_generic
#        relaunches, the restore puts the mirror back, the runner resumes.
# Example:
#   bash scripts/cloud/babysit_generic.sh bon scripts/cloud/run_bon.sh "=== BON done ===" \
#        "study_bon_clip_matrix.json study_bon_pickscore_matrix.json" \
#        "outputs/checkpoints/flowgrpo_clip_pub_s42 outputs/checkpoints/flowgrpo_pickscore_pub_s42" 12

set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
: "${LAMBDA_API_KEY:?}"; : "${HF_TOKEN:?}"

NAME="${1:?name}"; RUNNER="${2:?runner}"; SENTINEL="${3:?sentinel}"
EXPECT="${4:?expected outputs}"; UPLOADS="${5:-}"; WATCH_HOURS="${6:-14}"

KEY_NAME="claude-lambda-20260419-200846"
SSH_KEY="$HOME/.ssh/lambda_claude"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"
INST_FILE=".lambda_instance_g_$NAME"; IP_FILE=".lambda_ip_g_$NAME"
mkdir -p logs outputs
LOG="logs/babysit_g_$NAME.log"
log() { echo "[$(date +%H:%M:%S)][$NAME] $*" | tee -a "$LOG"; }
RUNNER_BASE=$(basename "${RUNNER%% *}")

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

SYNC="${BABYSIT_SYNC:-}"
REMOTE_ROOT="/home/ubuntu/haotian_research-1"

mirror_sync() {   # box -> local, paths preserved (rsync -R anchors at /./)
    local P
    for P in $SYNC; do
        rsync -azR --partial --exclude='*.tmp' -e "$SSH" \
            "ubuntu@$IP:$REMOTE_ROOT/./$P" ./ >> "$LOG" 2>&1 || log "  mirror: $P not synced (yet)"
    done
}
restore_sync() {  # local -> box, before a kick, so a resumed runner finds its checkpoints
    local P
    for P in $SYNC; do
        [ -e "$P" ] || continue
        log "restoring $P to the box ..."
        rsync -azR --partial -e "$SSH" "./$P" "ubuntu@$IP:$REMOTE_ROOT/" >> "$LOG" 2>&1 \
            || { log "FATAL: restore $P failed"; exit 1; }
    done
}
instance_status() {
    curl -s --max-time 30 -u "$LAMBDA_API_KEY:" "https://cloud.lambdalabs.com/api/v1/instances/$INST" \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('status') or ('GONE' if 'error' in d else 'QUERYFAIL'))" 2>/dev/null \
        || echo QUERYFAIL
}

log "=== generic driver start: $NAME ($RUNNER) ==="
if [ ! -f "$INST_FILE" ]; then
    CANDIDATES=$(curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instance-types | python3 -c "
import json, sys
d = json.load(sys.stdin)['data']
skip = set(__import__('os').environ.get('BABYSIT_SKIP_REGIONS', '').split())
for t in __import__('os').environ.get('BABYSIT_GPU_TYPES', 'gpu_1x_h100_pcie gpu_1x_h100_sxm5 gpu_1x_gh200').split():
    if t in d:
        for r in d[t]['regions_with_capacity_available']:
            if r['name'] not in skip:
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
    [ -f "$IP_FILE" ] || { log "not active after the wait; terminating the stuck instance"; terminate_box; log "FATAL: never active"; exit 1; }
fi
IP="$(cat "$IP_FILE")"
log "instance $INST @ $IP"

SSH_OK=0
for i in $(seq 1 40); do
    if $SSH "ubuntu@$IP" 'echo ok' >/dev/null 2>&1; then SSH_OK=1; log "SSH up."; break; fi
    sleep 15
done
[ "$SSH_OK" = 1 ] || { log "FATAL: no SSH"; exit 1; }

RUNNER_LOG="logs/g_${NAME}_runner.log"
PGPAT=$(echo "$RUNNER_BASE" | sed 's/\.sh$/[.]sh/')
STATE=$($SSH "ubuntu@$IP" "grep -qF '$SENTINEL' ~/haotian_research-1/$RUNNER_LOG 2>/dev/null && echo DONE || (pgrep -f '$PGPAT' >/dev/null && echo RUNNING || echo IDLE)" 2>/dev/null || echo IDLE)
log "remote state: $STATE"

if [ "$STATE" = "IDLE" ]; then
    # One runner per box: never kick on top of a live training/eval process
    # (two evals on one 40GB box OOM'd each other and the box idled 8h).
    STRAY=$($SSH "ubuntu@$IP" "pgrep -fa 'train_draft.py|train_distillation.py|train_sd3.py|eval_study_matrix.py|generate_tfg_dataset.py|pdep_measure.py' | grep -v pgrep | head -3" 2>/dev/null || true)
    if [ -n "$STRAY" ]; then
        log "FATAL: box already has a live GPU job, refusing to kick a second runner:"
        echo "$STRAY" | tee -a "$LOG"
        exit 1
    fi
    $SSH "ubuntu@$IP" 'mkdir -p ~/haotian_research-1/logs ~/haotian_research-1/outputs/checkpoints'
    rsync -az -e "$SSH" --delete scripts/ "ubuntu@$IP:/home/ubuntu/haotian_research-1/scripts/" >> "$LOG" 2>&1
    rsync -az -e "$SSH" configs/ "ubuntu@$IP:/home/ubuntu/haotian_research-1/configs/" >> "$LOG" 2>&1
    for U in $UPLOADS; do
        log "uploading $U ..."
        rsync -az --partial -e "$SSH" "$U" \
            "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/checkpoints/" >> "$LOG" 2>&1 \
            || { log "FATAL: upload $U failed"; exit 1; }
    done
    restore_sync
    # The kick ssh session can hang for ~35 min even though the runner started
    # (see lambda-ops notes); bound it and trust the pgrep check below instead.
    $SSH "ubuntu@$IP" "cd ~/haotian_research-1 && mkdir -p logs && \
        setsid nohup env HF_TOKEN='$HF_TOKEN' bash $RUNNER \
        > $RUNNER_LOG 2>&1 < /dev/null & echo kicked" >> "$LOG" 2>&1 &
    KPID=$!
    for _i in $(seq 1 24); do sleep 5; kill -0 $KPID 2>/dev/null || break; done
    kill $KPID 2>/dev/null && log "kick ssh hung; killed it, verifying runner via pgrep" || true
    sleep 20
    ALIVE=$($SSH "ubuntu@$IP" "pgrep -f '$PGPAT' | head -1" 2>/dev/null || true)
    if [ -z "$ALIVE" ]; then
        log "FATAL: runner not alive — runner log tail:"
        $SSH "ubuntu@$IP" "tail -25 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null" 2>/dev/null | tee -a "$LOG"
        exit 1
    fi
    log "runner alive (pid $ALIVE)"
fi

LOST=0
if [ "$STATE" != "DONE" ]; then
    DEADLINE=$(( $(date +%s) + WATCH_HOURS * 3600 ))
    IDLE_POLLS=0; UNREACH=0
    while [ "$(date +%s)" -lt "$DEADLINE" ]; do
        sleep 300
        mirror_sync
        # FATAL can be buried under a traceback: scan 40 lines, not 6.
        TAIL=$($SSH "ubuntu@$IP" "tail -40 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null" 2>/dev/null || true)
        if echo "$TAIL" | grep -qF "$SENTINEL"; then log "runner finished."; break; fi
        if echo "$TAIL" | grep -qF "FATAL:"; then log "runner FATAL"; break; fi
        HB=$($SSH "ubuntu@$IP" "tail -1 ~/haotian_research-1/$RUNNER_LOG 2>/dev/null | tr -d '\r' | head -c 90" 2>/dev/null || echo unreachable)
        # Idle-GPU guard: a live box doing 0% GPU for 20 min is a dead or hung runner.
        UTIL=$($SSH "ubuntu@$IP" "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1" 2>/dev/null || echo "?")
        ALIVE_NOW=$($SSH "ubuntu@$IP" "pgrep -f '$PGPAT' | head -1" 2>/dev/null || echo "?")
        if [ "$UTIL" = "0" ]; then IDLE_POLLS=$((IDLE_POLLS + 1)); else IDLE_POLLS=0; fi
        log "hb: gpu=${UTIL}% $HB"
        # Lost box (preempted / went unhealthy): 3 unreachable polls, then ask the API.
        if [ "$HB" = "unreachable" ]; then UNREACH=$((UNREACH + 1)); else UNREACH=0; fi
        if [ "$UNREACH" -ge 3 ]; then
            IST=$(instance_status)
            log "box unreachable for 15 min; API status '$IST'"
            case "$IST" in
                active|booting|QUERYFAIL) ;;   # QUERYFAIL = probably our own network; keep waiting
                terminated|GONE)
                   log "FATAL: instance lost ($IST); mirror has $(ls -d $SYNC 2>/dev/null | wc -l | tr -d ' ') path(s)"
                   rm -f "$INST_FILE" "$IP_FILE"; LOST=1; break;;
                *) # unhealthy etc.: still billed until terminated (verified; files removed on success)
                   log "FATAL: instance lost ($IST); terminating it before relaunch"
                   terminate_box; LOST=1; break;;
            esac
        fi
        if [ -z "$ALIVE_NOW" ]; then log "runner process gone without sentinel — treating as FATAL"; break; fi
        if [ "$IDLE_POLLS" -ge 4 ]; then log "GPU idle for 20 min with runner alive — treating as FATAL"; break; fi
    done
fi

if [ "$LOST" = 1 ]; then
    if [ -f "$INST_FILE" ]; then log "instance release unconfirmed — NOT relaunching; check the dashboard"; exit 1; fi
    log "=== generic driver: instance lost; exit 3 (kick_generic relaunches and restores the mirror) ==="
    exit 3
fi

log "pulling..."
mirror_sync
OK=1
for f in $EXPECT; do
    for a in 1 2 3; do
        rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/$f" ./outputs/ >> "$LOG" 2>&1 \
            && { log "  $f ok"; break; }
        sleep 8
    done
    [ -f "outputs/$f" ] || OK=0
done
rsync -az -e "$SSH" "ubuntu@$IP:/home/ubuntu/haotian_research-1/$RUNNER_LOG" ./logs/ >> "$LOG" 2>&1 || true

if [ "$OK" = 1 ]; then terminate_box; else log "NOT terminating: expected outputs missing"; fi
log "=== generic driver done: $NAME ==="
