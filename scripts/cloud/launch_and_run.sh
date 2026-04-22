#!/bin/bash
# Launches a fresh Lambda A100 instance, bootstraps the environment,
# and starts the target script in a tmux session.
#
# Required env vars:
#   LAMBDA_API_KEY   — Lambda Labs API key
#   HF_TOKEN         — HuggingFace token (for SD3.5-M)
#   WANDB_API_KEY    — W&B API key
#   SSH_KEY_PATH     — path to .pem file (default: ~/Downloads/haotian-research.pem)
#   SSH_KEY_NAME     — name of the SSH key registered in Lambda dashboard
#   RUN_SCRIPT       — script to run after setup (default: scripts/cloud/run_flowgrpo.sh)

set -euo pipefail

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── Config ────────────────────────────────────────────────────────────────────
LAMBDA_API_KEY="${LAMBDA_API_KEY:?Must set LAMBDA_API_KEY}"
HF_TOKEN="${HF_TOKEN:?Must set HF_TOKEN}"
WANDB_API_KEY="${WANDB_API_KEY:?Must set WANDB_API_KEY}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/Downloads/haotian-research.pem}"
SSH_KEY_NAME="${SSH_KEY_NAME:?Must set SSH_KEY_NAME (name in Lambda dashboard)}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/cloud/run_flowgrpo.sh}"

INSTANCE_TYPE="gpu_1x_a100_sxm4"
REPO_URL="https://github.com/nihalgunu/haotian_research.git"
TMUX_SESSION="run"

API="https://cloud.lambdalabs.com/api/v1"
AUTH="Authorization: Bearer $LAMBDA_API_KEY"

chmod 600 "$SSH_KEY_PATH"

# ── Step 1: find an available region for A100 ─────────────────────────────────
log "Finding available A100 region..."
REGION=$(curl -sf -H "$AUTH" "$API/instance-types" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for name, info in data['data'].items():
    if name == '$INSTANCE_TYPE':
        regions = info.get('regions_with_capacity_available', [])
        if regions:
            print(regions[0]['name'])
            sys.exit(0)
print('')
")

if [ -z "$REGION" ]; then
    log "FATAL: No $INSTANCE_TYPE capacity available right now. Try again later."
    exit 1
fi
log "  Region: $REGION"

# ── Step 2: launch instance ───────────────────────────────────────────────────
log "Launching $INSTANCE_TYPE in $REGION..."
LAUNCH_RESP=$(curl -sf -X POST -H "$AUTH" -H "Content-Type: application/json" \
    "$API/instance-operations/launch" \
    -d "{
        \"region_name\": \"$REGION\",
        \"instance_type_name\": \"$INSTANCE_TYPE\",
        \"ssh_key_names\": [\"$SSH_KEY_NAME\"],
        \"quantity\": 1
    }")

INSTANCE_ID=$(echo "$LAUNCH_RESP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
ids = d.get('data', {}).get('instance_ids', [])
print(ids[0] if ids else '')
")

if [ -z "$INSTANCE_ID" ]; then
    log "FATAL: Launch failed. Response: $LAUNCH_RESP"
    exit 2
fi
log "  Instance ID: $INSTANCE_ID"

# Save instance ID for manual termination if needed
echo "$INSTANCE_ID" > .lambda_instance
log "  (Saved to .lambda_instance — run terminate.sh to kill)"

# ── Step 3: wait for instance to be active ────────────────────────────────────
log "Waiting for instance to become active..."
for i in $(seq 1 60); do
    STATUS=$(curl -sf -H "$AUTH" "$API/instances/$INSTANCE_ID" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data']['status'])")
    IP=$(curl -sf -H "$AUTH" "$API/instances/$INSTANCE_ID" | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'].get('ip',''))")

    if [ "$STATUS" = "active" ] && [ -n "$IP" ]; then
        log "  Instance active at $IP"
        break
    fi
    log "  Status: $STATUS (attempt $i/60)..."
    sleep 15
done

if [ "$STATUS" != "active" ]; then
    log "FATAL: Instance never became active after 15 min"
    exit 3
fi

echo "$IP" > .current_instance_ip

# ── Step 4: wait for SSH to be ready ─────────────────────────────────────────
log "Waiting for SSH to be ready..."
for i in $(seq 1 20); do
    if ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
        ubuntu@"$IP" "echo ok" &>/dev/null; then
        log "  SSH ready"
        break
    fi
    log "  SSH not ready yet (attempt $i/20)..."
    sleep 10
done

SSH="ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no ubuntu@$IP"

# ── Step 5: transfer bootstrap script ────────────────────────────────────────
log "Transferring bootstrap script..."
scp -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no \
    "$(dirname "$0")/bootstrap.sh" ubuntu@"$IP":~/bootstrap.sh
$SSH "chmod +x ~/bootstrap.sh"

# ── Step 6: run bootstrap + target script in tmux ────────────────────────────
log "Starting bootstrap + $RUN_SCRIPT in tmux session '$TMUX_SESSION'..."
$SSH "tmux new-session -d -s $TMUX_SESSION \
    \"HF_TOKEN='$HF_TOKEN' WANDB_API_KEY='$WANDB_API_KEY' \
      REPO_URL='$REPO_URL' RUN_SCRIPT='$RUN_SCRIPT' \
      bash ~/bootstrap.sh 2>&1 | tee ~/run.log; \
      echo EXIT_CODE=\$? >> ~/run.log\""

log ""
log "========================================"
log "Instance running: $IP"
log "Monitor with:"
log "  ssh -i $SSH_KEY_PATH ubuntu@$IP"
log "  tmux attach -t $TMUX_SESSION"
log "  tail -f ~/run.log"
log ""
log "Terminate when done:"
log "  LAMBDA_API_KEY=$LAMBDA_API_KEY bash scripts/cloud/terminate.sh"
log "========================================"
