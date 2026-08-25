#!/bin/bash
# Runs ON the box, invoked over ssh. Detaches run_study_reward.sh completely so
# the ssh session can close immediately.
#
# The one-liner form (`cd && nohup ... > log & echo`) does NOT detach: the
# backgrounded job is the whole `cd && nohup ...` compound, so bash forks a
# subshell that waits on the runner while holding the ssh session's stdout /
# stderr pipes — the session then hangs until TCP timeout (~35 min) and the
# driver sees "kick failed". Here the redirections bind to the entire simple
# command, the parent shell exits at once, and setsid detaches the process
# group from the (soon-dead) session.
#
# Usage: HF_TOKEN=... bash kick_study.sh <clip|clip_ocr|pickscore> <WALL_HOURS>

set -euo pipefail
KEY="${1:?usage: kick_study.sh <reward_key> <wall_hours>}"
HOURS="${2:-4}"
: "${HF_TOKEN:?HF_TOKEN required}"

cd "$(dirname "$0")/../.."
mkdir -p logs

setsid nohup env HF_TOKEN="$HF_TOKEN" FG_GROUP="${FG_GROUP:-8}" FG_NBPE="${FG_NBPE:-8}" \
    bash scripts/cloud/run_study_reward.sh "$KEY" "$HOURS" \
    > "logs/study_${KEY}_runner.log" 2>&1 < /dev/null &
echo "kicked pid $!"
