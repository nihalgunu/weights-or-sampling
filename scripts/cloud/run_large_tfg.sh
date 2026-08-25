#!/bin/bash
# Runs ON the box (x86 H100). Guidance-side SD3.5-LARGE replication, eval-only:
# arms = base, TFG rho20, TFG rho20+mu20 on the CLIP setting, 150 prompts x 2 seeds.
# Usage: HF_TOKEN=... bash scripts/cloud/run_large_tfg.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?}"
log ">>> deps"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet --upgrade --force-reinstall pillow
python3 -m pip install --quiet 'diffusers>=0.31,<0.40' 'transformers==4.49.0' \
    'peft>=0.12' 'accelerate>=0.33' sentencepiece protobuf safetensors torchvision tqdm
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" \
          "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$PROMPTS_FILE" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY
mkdir -p "$PROJECT_ROOT/outputs/empty_ckpts"
log ">>> Large TFG eval"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --base stabilityai/stable-diffusion-3.5-large \
    --reward clip --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/empty_ckpts" \
    --tfg-rhos 20 --tfg-combos 20:20 --stack-rhos "" \
    --seed-bases 0,1000 \
    --out "$PROJECT_ROOT/outputs/study_large_tfg_matrix.json" \
    || { log "FATAL: large tfg eval failed"; exit 1; }
log "=== LARGETFG done ==="
