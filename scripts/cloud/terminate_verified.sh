#!/bin/bash
# Terminate a Lambda instance and VERIFY it (a silent curl failure once leaked ~$79).
# Usage: bash scripts/cloud/terminate_verified.sh <instance_id | path-to-instance-file>
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
: "${LAMBDA_API_KEY:?}"
ARG="${1:?instance id or file}"
if [ -f "$ARG" ]; then ID="$(cat "$ARG")"; FILE="$ARG"; else ID="$ARG"; FILE=""; fi
for a in 1 2 3 4 5; do
  echo "[terminate] $ID attempt $a"
  curl -s --max-time 30 -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" \
    -d "{\"instance_ids\":[\"$ID\"]}" https://cloud.lambdalabs.com/api/v1/instance-operations/terminate >/dev/null 2>&1
  sleep 15
  ST=$(curl -s --max-time 30 -u "$LAMBDA_API_KEY:" "https://cloud.lambdalabs.com/api/v1/instances/$ID" \
       | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('status') or ('GONE' if 'error' in d else 'QUERYFAIL'))" 2>/dev/null || echo QUERYFAIL)
  echo "[terminate] status: $ST"
  case "$ST" in terminating|terminated|GONE) [ -n "$FILE" ] && rm -f "$FILE" "${FILE/instance/ip}"; echo "[terminate] confirmed"; exit 0;; esac
done
echo "[terminate] ERROR: unconfirmed termination of $ID — CHECK DASHBOARD"; exit 1
