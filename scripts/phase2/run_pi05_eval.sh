#!/bin/bash
# Phase 2 M0: evaluate a PyTorch pi0.5 checkpoint on a LIBERO suite with RLinf's
# standalone eval script, inside the RLinf Docker image. Runs on a Lambda box.
#
# Usage: bash scripts/phase2/run_pi05_eval.sh [SUITE] [TRIALS_PER_TASK] [HF_REPO] [TAG]
#   SUITE   libero_spatial | libero_object | libero_goal | libero_10   (default libero_spatial)
#   TRIALS  episodes per task (default 10; paper number is 50)
#   HF_REPO default RLinf/RLinf-Pi05-LIBERO-SFT
# Outputs: outputs/phase2/pi05_eval_<SUITE><TAG>.json, outputs/phase2/logs/<exp>/..., sentinel line
#   "=== PHASE2 EVAL done ===" on success.
set -uo pipefail
cd "$(dirname "$0")/../.."
SUITE="${1:-libero_spatial}"; TRIALS="${2:-10}"; HF_REPO="${3:-RLinf/RLinf-Pi05-LIBERO-SFT}"; TAG="${4:-}"
IMG="rlinf/rlinf:agentic-rlinf0.4-maniskill_libero"
RLINF_DIR="$PWD/repos/RLinf"; MODELS="$PWD/models"; OUT="$PWD/outputs/phase2"
EXP="pi05_sft_${SUITE}${TAG}"
log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$MODELS" "$OUT/logs" repos

log "=== PHASE2 EVAL start: $SUITE x $TRIALS trials/task, $HF_REPO ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
docker --version || { log "FATAL: docker missing"; exit 1; }

# 1. code + weights (host side, cached across runs)
[ -d "$RLINF_DIR/.git" ] || git clone --depth 1 https://github.com/RLinf/RLinf.git "$RLINF_DIR" || { log "FATAL: clone RLinf"; exit 1; }
CKPT="$MODELS/$(basename "$HF_REPO")"
if [ ! -f "$CKPT/model.safetensors" ]; then
  python3 -m pip install --quiet 'huggingface_hub[cli]' >/dev/null 2>&1
  log "downloading $HF_REPO ..."
  python3 - "$HF_REPO" "$CKPT" <<'PY' || { echo "FATAL: HF download"; exit 1; }
import sys, os
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], token=os.environ.get("HF_TOKEN"))
PY
fi
log "checkpoint tree:"; find "$CKPT" -maxdepth 3 | head -30; du -sh "$CKPT"
# openpi loads norm stats from <ckpt>/assets/<asset_id>/norm_stats.json; the HF repo ships
# the asset folder at the root. Make both layouts available.
if [ -d "$CKPT/physical-intelligence" ] && [ ! -d "$CKPT/assets/physical-intelligence" ]; then
  mkdir -p "$CKPT/assets" && cp -r "$CKPT/physical-intelligence" "$CKPT/assets/" && log "mirrored assets/ layout"
fi

# 2. image
docker image inspect "$IMG" >/dev/null 2>&1 || { log "pulling $IMG ..."; docker pull "$IMG" || { log "FATAL: docker pull"; exit 1; }; }

# 3. eval inside the container
log ">>> eval $EXP"
docker run --rm --gpus all --shm-size 32g --ipc=host \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$RLINF_DIR":/workspace/RLinf -v "$MODELS":/models -v "$OUT":/outputs \
  "$IMG" bash -lc "
    set -o pipefail
    if command -v switch_env >/dev/null 2>&1; then source switch_env openpi; else echo 'no switch_env on PATH; using image default env'; fi
    cd /workspace/RLinf
    export REPO_PATH=/workspace/RLinf PYTHONPATH=/workspace/RLinf:\${PYTHONPATH:-}
    python toolkits/standalone_eval_scripts/openpi/libero_eval.py \
      --log_dir /outputs/logs --exp_name $EXP --config_name pi05_libero \
      --pretrained_path /models/$(basename "$HF_REPO") --task_suite_name $SUITE \
      --num_trials_per_task $TRIALS --action_chunk 5 --num_steps 5 \
      --num_save_videos 5 --video_temp_subsample 10 --seed 7
  " 2>&1 | tee "$OUT/logs/${EXP}.stdout"
RC=${PIPESTATUS[0]}
[ "$RC" = 0 ] || { log "FATAL: eval exited $RC"; exit 1; }

# 4. summary json from the log
python3 - "$OUT/logs/${EXP}.stdout" "$OUT/pi05_eval_${SUITE}${TAG}.json" "$SUITE" "$TRIALS" "$HF_REPO" <<'PY'
import re, sys, json
txt = open(sys.argv[1], errors="replace").read()
tasks = re.findall(r"Task: (.+?), Successes: (\d+)/(\d+), Success Rate: ([\d.]+)", txt)
tot = re.search(r"total_success_rate[^\d]*([\d.]+)", txt)
eps = re.search(r"total_episodes[^\d]*(\d+)", txt)
out = {"suite": sys.argv[3], "trials_per_task": int(sys.argv[4]), "checkpoint": sys.argv[5],
       "num_steps": 5, "action_chunk": 5, "seed": 7,
       "total_success_rate": float(tot.group(1)) if tot else None,
       "total_episodes": int(eps.group(1)) if eps else None,
       "per_task": [{"task": t, "succ": int(s), "n": int(n), "rate": float(r)} for t, s, n, r in tasks]}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("summary:", out["total_success_rate"], "over", out["total_episodes"], "episodes;", len(tasks), "tasks")
PY
log "=== PHASE2 EVAL done ==="
