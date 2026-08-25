#!/bin/bash
# Provision a Lambda instance and wait until it's SSH-able.
# Writes .lambda_instance (id) and .lambda_ip (IPv4). terminate.sh consumes the id.
#
# Usage: bash scripts/cloud/launch_lambda.sh <instance_type> <region> <ssh_key_name>
# Example: bash scripts/cloud/launch_lambda.sh gpu_1x_a100_sxm4 us-east-1 claude-lambda-20260419

set -euo pipefail

INSTANCE_TYPE="${1:?usage: launch_lambda.sh <instance_type> <region> <ssh_key_name>}"
REGION="${2:?missing region}"
SSH_KEY_NAME="${3:?missing ssh_key_name}"

: "${LAMBDA_API_KEY:?set LAMBDA_API_KEY (source .env)}"

cd "$(dirname "$0")/../.."

if [ -f .lambda_instance ]; then
    echo "ERROR: .lambda_instance already exists (id=$(cat .lambda_instance))."
    echo "  Terminate first: bash scripts/cloud/terminate.sh"
    exit 1
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "Launching $INSTANCE_TYPE in $REGION with key $SSH_KEY_NAME..."
LAUNCH_BODY=$(cat <<JSON
{"region_name":"$REGION","instance_type_name":"$INSTANCE_TYPE","ssh_key_names":["$SSH_KEY_NAME"],"quantity":1}
JSON
)

RESP=$(curl -s -u "$LAMBDA_API_KEY:" \
    -H "Content-Type: application/json" \
    -X POST \
    -d "$LAUNCH_BODY" \
    https://cloud.lambdalabs.com/api/v1/instance-operations/launch)

ID=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('instance_ids',[''])[0] or '')")

if [ -z "$ID" ]; then
    echo "ERROR: launch failed:"
    echo "$RESP" | python3 -m json.tool
    exit 1
fi

echo "$ID" > .lambda_instance
log "Instance id: $ID"
log "Polling until status=active (this usually takes 1-3 min)..."

for i in $(seq 1 60); do
    sleep 10
    INFO=$(curl -s -u "$LAMBDA_API_KEY:" "https://cloud.lambdalabs.com/api/v1/instances/$ID")
    STATUS=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('status','?'))")
    IP=$(echo "$INFO"     | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('ip','') or '')")
    log "  attempt $i: status=$STATUS ip=${IP:-pending}"
    if [ "$STATUS" = "active" ] && [ -n "$IP" ]; then
        echo "$IP" > .lambda_ip
        log "Instance active at $IP"
        log "  ssh -i ~/.ssh/lambda_claude -o StrictHostKeyChecking=no ubuntu@$IP"
        log "  terminate when done: bash scripts/cloud/terminate.sh"
        exit 0
    fi
done

log "ERROR: instance did not reach active state within 10 min. Check the dashboard."
exit 1
