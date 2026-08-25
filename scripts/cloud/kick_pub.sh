#!/bin/bash
# Runs ON the box, invoked over ssh. Fully detaches run_pub_reward.sh (see
# kick_study.sh for why the one-liner form does not detach).
# Usage: HF_TOKEN=... bash kick_pub.sh <clip|clip_ocr|pickscore> <WALL_HOURS_PER_SEED>

set -euo pipefail
KEY="${1:?usage: kick_pub.sh <reward_key> <wall_hours_per_seed>}"
HOURS="${2:-4}"
: "${HF_TOKEN:?HF_TOKEN required}"

cd "$(dirname "$0")/../.."
mkdir -p logs

setsid nohup env HF_TOKEN="$HF_TOKEN" FG_GROUP=8 FG_NBPE=8 \
    bash scripts/cloud/run_pub_reward.sh "$KEY" "$HOURS" \
    > "logs/pub_${KEY}_runner.log" 2>&1 < /dev/null &
echo "kicked pid $!"
