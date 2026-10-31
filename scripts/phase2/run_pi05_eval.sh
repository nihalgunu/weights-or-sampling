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
# Lambda's ubuntu user is not in the docker group; fall back to passwordless sudo.
if docker info >/dev/null 2>&1; then DOCKER="docker"; elif sudo -n docker info >/dev/null 2>&1; then DOCKER="sudo -n docker"; else log "FATAL: docker daemon not reachable (even with sudo)"; exit 1; fi
log "docker via: $DOCKER"

# 1. code + weights (host side, cached across runs)
[ -d "$RLINF_DIR/.git" ] || git clone --depth 1 https://github.com/RLinf/RLinf.git "$RLINF_DIR" || { log "FATAL: clone RLinf"; exit 1; }
CKPT="$MODELS/$(basename "$HF_REPO")"
if [ ! -f "$CKPT/model.safetensors" ]; then
  log "downloading $HF_REPO ..."
  # Lambda's system python ships an old zstandard that breaks huggingface_hub's vendored
  # httpx decoder ("process() takes no keyword arguments"). Use a clean venv first,
  # fall back to git-lfs.
  python3 -m venv "$PWD/.venv_hf" >/dev/null 2>&1 && "$PWD/.venv_hf/bin/pip" install --quiet --upgrade pip 'huggingface_hub>=0.30' zstandard >/dev/null 2>&1
  "$PWD/.venv_hf/bin/python" - "$HF_REPO" "$CKPT" <<'PY' || DL_FAIL=1
import sys, os
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], token=os.environ.get("HF_TOKEN"))
PY
  if [ "${DL_FAIL:-0}" = 1 ] || [ ! -f "$CKPT/model.safetensors" ]; then
    log "snapshot_download failed; falling back to git-lfs clone"
    rm -rf "$CKPT"
    (command -v git-lfs >/dev/null || (sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs)) && git lfs install --skip-repo
    GIT_LFS_SKIP_SMUDGE=0 git clone "https://${HF_TOKEN:+user:$HF_TOKEN@}huggingface.co/$HF_REPO" "$CKPT" || { log "FATAL: HF download (both methods)"; exit 1; }
    [ -f "$CKPT/model.safetensors" ] || { log "FATAL: model.safetensors missing after clone"; exit 1; }
  fi
fi
# 2. image (pulled before provenance so the digest is recorded)
$DOCKER image inspect "$IMG" >/dev/null 2>&1 || { log "pulling $IMG ..."; $DOCKER pull "$IMG" || { log "FATAL: docker pull"; exit 1; }; }
# openpi loads <ckpt>/<asset_id>/norm_stats.json with asset_id physical-intelligence/libero.
# The big file can be present while this small one is missing (first run's decoder bug), so
# check it separately and fetch it directly if needed.
NS="$CKPT/physical-intelligence/libero/norm_stats.json"
if [ ! -s "$NS" ]; then
  log "norm_stats.json missing; fetching directly"
  mkdir -p "$(dirname "$NS")"
  curl -fsSL --retry 3 ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
    "https://huggingface.co/$HF_REPO/resolve/main/physical-intelligence/libero/norm_stats.json" -o "$NS" \
    || { log "FATAL: could not fetch norm_stats.json"; exit 1; }
fi
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('norm_stats keys:', list(d)[:6])" "$NS" || { log "FATAL: norm_stats.json unreadable"; exit 1; }
log "checkpoint tree:"; find "$CKPT" -maxdepth 3 | head -30; du -sh "$CKPT"
# Integrity: the safetensors header must parse and the file must not be an LFS pointer.
python3 - "$CKPT/model.safetensors" <<'PY' || { log "FATAL: checkpoint integrity"; exit 1; }
import sys, os, json, struct, hashlib
p = sys.argv[1]; sz = os.path.getsize(p)
assert sz > 1_000_000_000, f"model.safetensors is only {sz} bytes (LFS pointer?)"
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n))
tensors = [k for k in hdr if k != "__metadata__"]
print(f"safetensors ok: {sz/1e9:.2f} GB, {len(tensors)} tensors, e.g. {tensors[:3]}")
PY
PROV="$OUT/provenance_$(date +%Y%m%d_%H%M%S).json"
python3 - "$PROV" "$RLINF_DIR" "$CKPT" "$IMG" "$DOCKER" <<'PY'
import sys, json, subprocess, os, hashlib
prov = {"rlinf_commit": subprocess.run(["git", "-C", sys.argv[2], "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "image": sys.argv[4],
        "image_digest": subprocess.run(sys.argv[5].split() + ["image", "inspect", "--format", "{{index .RepoDigests 0}}", sys.argv[4]], capture_output=True, text=True).stdout.strip() or None,
        "ckpt_size": os.path.getsize(os.path.join(sys.argv[3], "model.safetensors")),
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()}
json.dump(prov, open(sys.argv[1], "w"), indent=1); print("provenance:", prov)
PY


# 3. eval inside the container
log ">>> eval $EXP"
$DOCKER run --rm --gpus all --shm-size 32g --ipc=host \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$RLINF_DIR":/workspace/RLinf -v "$MODELS":/models -v "$OUT":/outputs \
  "$IMG" bash -lc "
    set -o pipefail
    if command -v switch_env >/dev/null 2>&1; then source switch_env openpi; else echo 'no switch_env on PATH; using image default env'; fi
    cd /workspace/RLinf
    export REPO_PATH=/workspace/RLinf PYTHONPATH=/workspace/RLinf:\${PYTHONPATH:-}
    echo \"container python: \$(which python) \$(python --version 2>&1)\"
    python -c 'import torch, openpi, libero; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"openpi\", getattr(openpi, \"__version__\", \"?\"))' || { echo \"FATAL: container env missing torch/openpi/libero\"; exit 2; }
    [ -f toolkits/standalone_eval_scripts/openpi/libero_eval.py ] || { echo \"FATAL: libero_eval.py not in RLinf checkout\"; exit 2; }
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
assert out["total_success_rate"] is not None and out["total_episodes"] == int(sys.argv[4]) * len(tasks) and len(tasks) == 10, "incomplete eval"
PY
log "=== PHASE2 EVAL done ==="
