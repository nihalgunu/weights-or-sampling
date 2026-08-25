#!/bin/bash
# Runs ON the box. mu-term ablation (eval-only, CLIP setting, 150 prompts):
# arms = base, tfg_rho20, tfg_mu5, tfg_mu20, tfg_rho20_mu20.
# Usage: HF_TOKEN=... bash scripts/cloud/run_mu.sh

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?HF_TOKEN required}"

log "=== MUEVAL: deps ==="
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet --upgrade --force-reinstall pillow
python3 -m pip install --quiet 'diffusers>=0.31,<0.40' 'transformers==4.49.0' \
    'peft>=0.12' 'accelerate>=0.33' sentencepiece protobuf safetensors torchvision tqdm
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"

PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" \
          "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$PROMPTS_FILE" <<'PY2'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY2

mkdir -p "$PROJECT_ROOT/outputs/empty_ckpts"
log "=== MUEVAL: matrix ==="
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward clip \
    --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/empty_ckpts" \
    --tfg-rhos 20 --tfg-mus 5,20 --tfg-combos 20:20 --stack-rhos "" \
    --seed-bases 0,1000,2000 \
    --out "$PROJECT_ROOT/outputs/study_mu_matrix.json"

log "=== MUEVAL done ==="
