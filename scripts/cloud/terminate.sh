#!/bin/bash
# Kill switch. Reads instance id from .lambda_instance (written by launch).
# Usage: bash scripts/cloud/terminate.sh [instance_id]
# If no id passed, reads from .lambda_instance.

set -euo pipefail

cd "$(dirname "$0")/../.."

: "${LAMBDA_API_KEY:?set LAMBDA_API_KEY to run this script}"

ID="${1:-}"
if [ -z "$ID" ] && [ -f .lambda_instance ]; then
    ID=$(cat .lambda_instance | head -1)
fi
if [ -z "$ID" ]; then
    echo "ERROR: no instance id provided and .lambda_instance missing"
    exit 1
fi

echo "Terminating instance $ID ..."
curl -s -u "$LAMBDA_API_KEY:" \
    -H "Content-Type: application/json" \
    -d "{\"instance_ids\": [\"$ID\"]}" \
    https://cloud.lambdalabs.com/api/v1/instance-operations/terminate | python3 -m json.tool

rm -f .lambda_instance
echo "Termination request sent. Verify with:"
echo "  curl -s -u \"\$LAMBDA_API_KEY:\" https://cloud.lambdalabs.com/api/v1/instances | python3 -m json.tool"
