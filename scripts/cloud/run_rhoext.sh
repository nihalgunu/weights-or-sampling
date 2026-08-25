#!/bin/bash
# Runs ON the box. Eval-only extension: TFG-on-PickScore at rho {40, 80} —
# tests whether the PickScore rho-response (still rising at rho=20) saturates,
# reverses, or keeps climbing toward RL's +0.0160. No training, no LoRA arms.
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_rhoext.sh

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?HF_TOKEN required}"

log "=== RHOEXT: pip deps ==="
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet --upgrade --force-reinstall pillow
python3 -m pip install --quiet 'diffusers>=0.31,<0.40' 'transformers==4.49.0' \
    'peft>=0.12' 'accelerate>=0.33' sentencepiece protobuf safetensors torchvision tqdm
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"

PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_ddrl.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" "$PROMPTS_FILE" <<'PY'
import json, sys, pathlib
src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
prompts = [d["prompt"] for d in json.loads(src.read_text())]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(prompts) + "\n")
print(f"wrote {len(prompts)} prompts -> {out}")
PY

mkdir -p "$PROJECT_ROOT/outputs/empty_ckpts"
log "=== RHOEXT: eval matrix (tfg rho 40,80 on pickscore) ==="
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward pickscore \
    --prompts-file "$PROMPTS_FILE" \
    --num-prompts 30 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/empty_ckpts" \
    --tfg-rhos ${RHOEXT_RHOS:-80} --stack-rhos "" \
    --seed-bases 0,1000,2000 \
    --image-dir "$PROJECT_ROOT/outputs/study_rhoext80_images" \
    --out "$PROJECT_ROOT/outputs/study_pickscore_rhoext80_matrix.json"

log "=== RHOEXT done ==="
