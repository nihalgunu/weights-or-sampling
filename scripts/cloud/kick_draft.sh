#!/bin/bash
# Local launcher for the DRaFT arm. Starts babysit_generic.sh DETACHED (the
# driver launches a Lambda 80GB box, syncs scripts/, kicks the runner, polls,
# pulls the expected files, and terminates the box only when every expected
# file arrived).
#
# Usage:
#   bash scripts/cloud/kick_draft.sh pilot [PILOT_MINUTES]            # ~2h,  ~$7
#   bash scripts/cloud/kick_draft.sh <clip|clip_ocr|pickscore|aesthetic> [WALL_HOURS] [LR]   # 3 x 4h + eval, ~15h, ~$50
#
# Needs .env with LAMBDA_API_KEY and HF_TOKEN. Watch: tail -f logs/babysit_g_draft_<key>.log
set -euo pipefail
cd "$(dirname "$0")/../.."
KEY="${1:?usage: kick_draft.sh <pilot|clip|clip_ocr|pickscore|aesthetic> [hours|minutes] [lr]}"
mkdir -p logs
if [ "$KEY" = "pilot" ]; then
  MIN="${2:-25}"
  NAME="draft_pilot"
  RUNNER="scripts/cloud/run_draft_pilot.sh $MIN"
  SENTINEL="=== DRaFT pilot done"
  EXPECT="study_draft_pilot_matrix.json draft_pilot_logs.tar.gz"
  WATCH=5
else
  HOURS="${2:-4}"; LR="${3:-1e-4}"; SEEDS="${4:-42 1042 2042}"
  SEEDBASES="${5:-0,1000,2000}"; CKPTARMS="${6:-final,best}"; TAG="${7:-}"; SAVEMIN="${8:-30}"; KL="${9:-0}"
  NAME="draft_${KEY}${TAG}${KICK_SUFFIX:-}"
  RUNNER="scripts/cloud/run_draft_reward.sh $KEY $HOURS $LR '$SEEDS' $SEEDBASES $CKPTARMS '$TAG' $SAVEMIN $KL"
  SENTINEL="=== DRAFT[$KEY$TAG] done ==="
  EXPECT="study_${KEY}_draft${TAG}_matrix.json study_${KEY}_draft${TAG}_external.json study_${KEY}_draft${TAG}_images.tar.gz draft_${KEY}${TAG}_ckpts.tar.gz"
  WATCH=$(python3 -c "print(int(3*$HOURS + 6))")
fi
# macOS has no setsid; nohup + disown detaches well enough for a local driver.
if command -v setsid >/dev/null 2>&1; then
  setsid nohup bash scripts/cloud/babysit_generic.sh "$NAME" "$RUNNER" "$SENTINEL" "$EXPECT" "" "$WATCH" \
    > "logs/kick_${NAME}.out" 2>&1 < /dev/null &
else
  nohup bash scripts/cloud/babysit_generic.sh "$NAME" "$RUNNER" "$SENTINEL" "$EXPECT" "" "$WATCH" \
    > "logs/kick_${NAME}.out" 2>&1 < /dev/null &
  disown
fi
echo "kicked $NAME (pid $!) — log: logs/babysit_g_${NAME}.log"
