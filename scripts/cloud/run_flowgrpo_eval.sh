#!/bin/bash
# Runs ON the Lambda instance. Sets up minimal deps for SD3.5 inference + LoRA
# eval, then runs eval_lora_compressibility.py against the rsync'd checkpoints.
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_flowgrpo_eval.sh

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ -z "${HF_TOKEN:-}" ]; then
    log "ERROR: HF_TOKEN not set. SD3.5-M is gated."
    exit 1
fi

log "=== FlowGRPO LoRA eval starting ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

# Clone flow_grpo only for the prompts file (dataset/ocr/test.txt).
FLOW_GRPO_DIR="$PROJECT_ROOT/repos/flow_grpo"
if [ ! -d "$FLOW_GRPO_DIR" ]; then
    log "cloning flow_grpo for prompts dataset..."
    mkdir -p "$PROJECT_ROOT/repos"
    git clone --depth 1 https://github.com/yifan123/flow_grpo "$FLOW_GRPO_DIR"
fi

log "installing minimal deps (diffusers, peft, transformers)..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet --upgrade --force-reinstall pillow
python3 -m pip install --quiet \
    'diffusers>=0.31,<0.40' \
    'transformers==4.49.0' \
    'peft>=0.12' \
    'accelerate>=0.33' \
    sentencepiece \
    protobuf \
    safetensors

log "HF login..."
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"

log "launching eval (30 prompts × 4 endpoints × 40 steps, ~25 min)..."
mkdir -p "$PROJECT_ROOT/outputs"
python3 "$PROJECT_ROOT/scripts/cloud/eval_lora_compressibility.py" \
    --base stabilityai/stable-diffusion-3.5-medium \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints/flowgrpo" \
    --prompts-file "$FLOW_GRPO_DIR/dataset/ocr/test.txt" \
    --num-prompts 30 \
    --num-steps 40 \
    --out "$PROJECT_ROOT/outputs/flowgrpo_eval.json"

log "=== FlowGRPO LoRA eval done ==="
log "Result: outputs/flowgrpo_eval.json"
