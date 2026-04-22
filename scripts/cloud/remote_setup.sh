#!/bin/bash
# Runs on the Lambda instance. Prereqs: Lambda Stack (torch + CUDA) pre-installed.
# Reads HF_TOKEN from env; caller (local orchestrator) sets it via ssh -o SendEnv
# or as a literal in the remote invocation. HF_TOKEN is required for SD3.5-M.

set -euo pipefail

cd "$(dirname "$0")/../.."

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ -z "${HF_TOKEN:-}" ]; then
    log "ERROR: HF_TOKEN not set. Cannot download gated SD3.5-M."
    exit 1
fi

log "=== Remote setup starting ==="
log "Python: $(python3 --version)"
log "CUDA visible: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())')"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

# Fresh venv isolates our pins from Lambda Stack; torch stays system-wide.
log "Installing Python deps..."
python3 -m pip install --quiet --upgrade pip
# --force-reinstall on pillow: Lambda ships Pillow 9.0.1 system-wide, which is
# missing PIL.Image.Resampling (added in 9.1). Modern transformers imports that
# attribute at load time, and a plain --user pillow install doesn't reliably
# shadow the system version. Force the user-site wheel to be installed.
python3 -m pip install --quiet --upgrade --force-reinstall pillow
python3 -m pip install --quiet \
    'diffusers>=0.31,<0.40' \
    'transformers>=4.44,<4.60' \
    'peft>=0.12' \
    'accelerate>=0.33' \
    sentencepiece \
    protobuf \
    safetensors \
    pyyaml \
    tqdm \
    huggingface_hub

log "HF login..."
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"

log "Pre-downloading SD3.5-M (~20 GB)..."
python3 <<'PY'
import torch, time
from diffusers import StableDiffusion3Pipeline
t0 = time.time()
_ = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16,
)
print(f"SD3.5-M downloaded in {time.time()-t0:.0f}s")
PY

log "Pre-downloading CLIP-L/14 for predictor..."
python3 <<'PY'
from transformers import CLIPModel, CLIPTokenizer
CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
print("CLIP cached")
PY

log "=== SETUP DONE ==="
