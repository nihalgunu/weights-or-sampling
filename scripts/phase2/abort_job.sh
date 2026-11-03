#!/bin/bash
# Abort a Phase 2 eval job: stop the local driver, kill the runner/container on the box, pull logs,
# terminate the instance with verification. Usage: bash scripts/phase2/abort_job.sh NAME
set -uo pipefail
cd "$(dirname "$0")/../.."
N="${1:?job name, e.g. p2_eval_libero_spatial_base50}"; INST=".lambda_instance_g_$N"
pkill -f "kick_${N}.sh" 2>/dev/null; pkill -f "babysit_generic.sh $N" 2>/dev/null; echo "[abort] local driver stopped"
IP=$(grep -oE "launched [0-9a-f]+ .*" "logs/babysit_g_$N.log" >/dev/null; set -a; source .env; set +a; curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instances | python3 -c "import json,sys,re; ids=open('$INST').read().split() if __import__('os').path.exists('$INST') else []; [print(i['ip']) for i in json.load(sys.stdin)['data'] if not ids or i['id'] in ids or i['id'][:8] in ' '.join(ids)]" | head -1)
K="$HOME/.ssh/lambda_claude"
if [ -n "$IP" ]; then
  ssh -i "$K" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "ubuntu@$IP" "pkill -f run_pi05_eval.sh; (sudo -n docker ps -q | xargs -r sudo -n docker kill) 2>/dev/null; echo killed" || true
  rsync -az -e "ssh -i $K -o StrictHostKeyChecking=no" "ubuntu@$IP:/home/ubuntu/haotian_research-1/outputs/phase2/logs/" outputs/phase2/logs/ 2>/dev/null || true
fi
[ -f "$INST" ] && bash scripts/cloud/terminate_verified.sh "$INST" || echo "[abort] no instance file $INST — check the Lambda API"
