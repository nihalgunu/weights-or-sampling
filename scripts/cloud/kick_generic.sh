#!/bin/bash
# Local detached launcher for babysit_generic.sh, with a capacity retry loop.
# Usage: bash scripts/cloud/kick_generic.sh NAME "RUNNER CMD" "SENTINEL" "EXPECT FILES" WATCH_HOURS [RETRY_HOURS]
# Env:   BABYSIT_GPU_TYPES="gpu_1x_h100_pcie gpu_1x_h100_sxm5 gpu_1x_gh200" (default); add gpu_1x_a100_sxm4 for 40GB-ok jobs.
#        BABYSIT_SYNC="outputs/checkpoints/<run> ..." paths mirrored every poll and restored on relaunch
#        (lost box -> babysit exits 3 -> relaunch within RETRY_HOURS -> runner resumes from the mirror).
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME="${1:?}"; RUNNER="${2:?}"; SENTINEL="${3:?}"; EXPECT="${4:?}"; WATCH="${5:?}"; RETRY_HOURS="${6:-6}"; GRACE_MIN="${GRACE_MIN:-30}"
mkdir -p logs
cat > "logs/kick_${NAME}.sh" <<WRAP
#!/bin/bash
cd "$(pwd)"
export BABYSIT_GPU_TYPES="${BABYSIT_GPU_TYPES:-gpu_1x_h100_pcie gpu_1x_h100_sxm5 gpu_1x_gh200}"
export BABYSIT_SKIP_REGIONS="${BABYSIT_SKIP_REGIONS:-}"
export BABYSIT_SYNC="${BABYSIT_SYNC:-}"
export BABYSIT_IDLE_POLLS="${BABYSIT_IDLE_POLLS:-4}"
DEADLINE=\$(( \$(date +%s) + ${RETRY_HOURS} * 3600 ))
LOSSES=0
while :; do
  bash scripts/cloud/babysit_generic.sh "$NAME" "$RUNNER" "$SENTINEL" "$EXPECT" "" "$WATCH"
  RC=\$?
  if [ "\$RC" = 3 ]; then
    LOSSES=\$((LOSSES + 1))
    [ "\$LOSSES" -le "${MAX_LOSSES:-5}" ] || { echo "[kick] box lost \$LOSSES times; giving up"; exit 1; }
    [ "\$(date +%s)" -lt "\$DEADLINE" ] || { echo "[kick] box lost after the retry deadline; giving up"; exit 1; }
    echo "[kick] box lost (\$LOSSES); relaunching in 2 min and restoring the mirror"; sleep 120; continue
  fi
  if tail -3 "logs/babysit_g_${NAME}.log" | grep -qE "FATAL: (no 80GB capacity|all refused|never active)"; then
    [ "\$(date +%s)" -lt "\$DEADLINE" ] || { echo "[kick] gave up waiting for capacity"; exit 1; }
    echo "[kick] no capacity, retry in 10 min"; sleep 600; continue
  fi
  break
done
# Unattended safety: the babysitter keeps a box alive after a runner FATAL (for
# debugging). Give a human ${GRACE_MIN} minutes to intervene, then terminate
# with verification so a failed run cannot leak a box overnight.
if grep -q "NOT terminating" "logs/babysit_g_${NAME}.log" 2>/dev/null && [ -f ".lambda_instance_g_${NAME}" ]; then
  echo "[kick] runner failed; auto-terminating in ${GRACE_MIN} min unless .lambda_instance_g_${NAME} is removed"
  sleep $(( ${GRACE_MIN} * 60 ))
  [ -f ".lambda_instance_g_${NAME}" ] && bash scripts/cloud/terminate_verified.sh ".lambda_instance_g_${NAME}"
fi
WRAP
if command -v setsid >/dev/null 2>&1; then
  setsid nohup bash "logs/kick_${NAME}.sh" > "logs/kick_${NAME}.out" 2>&1 < /dev/null &
else
  nohup bash "logs/kick_${NAME}.sh" > "logs/kick_${NAME}.out" 2>&1 < /dev/null &
  disown
fi
echo "kicked $NAME (pid $!) — log: logs/babysit_g_${NAME}.log"
