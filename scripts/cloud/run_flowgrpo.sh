#!/bin/bash
# Runs ON the Lambda instance. Sets up FlowGRPO and trains a mini run on
# SD3.5-M with the GenEval reward in LoRA mode, capped to a wall-clock budget.
#
# Pre-reqs on the box: Lambda Stack (torch + CUDA), HF_TOKEN env var set,
# and this repo rsynced to ~/haotian_research-1.
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_flowgrpo.sh [WALL_HOURS]
# Default WALL_HOURS=5 (leaves an hour for setup + eval inside a 6h budget).

set -euo pipefail

WALL_HOURS="${1:-5}"
WALL_SECONDS=$(python3 -c "print(int(float('$WALL_HOURS')*3600))")

# pip --user installs (accelerate, etc.) land in ~/.local/bin, which isn't on
# PATH for non-interactive ssh shells.
export PATH="$HOME/.local/bin:$PATH"

# train_sd3.py calls wandb.init unconditionally; we don't have a key on the
# box and don't need cloud logging for this mini run.
export WANDB_MODE=disabled

# Reduce memory fragmentation on 40GB A100 — without this we OOM at first step.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ -z "${HF_TOKEN:-}" ]; then
    log "ERROR: HF_TOKEN not set. SD3.5-M is gated."
    exit 1
fi

log "=== FlowGRPO mini run starting (budget: ${WALL_HOURS}h / ${WALL_SECONDS}s) ==="
log "Python: $(python3 --version)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

# --- Clone + pin upstream flow_grpo ---
FLOW_GRPO_COMMIT="main"  # pin to a SHA after first successful run
FLOW_GRPO_DIR="$PROJECT_ROOT/repos/flow_grpo"
mkdir -p "$PROJECT_ROOT/repos"
if [ ! -d "$FLOW_GRPO_DIR" ]; then
    log "Cloning flow_grpo..."
    git clone https://github.com/yifan123/flow_grpo "$FLOW_GRPO_DIR"
fi
cd "$FLOW_GRPO_DIR"
git fetch --quiet
git checkout --quiet "$FLOW_GRPO_COMMIT"
log "flow_grpo at $(git rev-parse --short HEAD)"

# --- Python deps (mirrors remote_setup.sh's pinning style) ---
log "Installing Python deps..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet --upgrade --force-reinstall pillow
python3 -m pip install --quiet \
    'diffusers>=0.31,<0.40' \
    'transformers==4.49.0' \
    'peft>=0.12' \
    'accelerate>=0.33' \
    'ml_collections' \
    'absl-py' \
    'einops' \
    'wandb' \
    sentencepiece \
    protobuf \
    safetensors \
    pyyaml \
    tqdm \
    huggingface_hub

# Reward = jpeg_compressibility — fully in-process (PIL.JPEG byte counts).
# No detector / no reward server needed. See flow_grpo/rewards.py:22.

# Install flow_grpo itself so train_sd3.py can `import flow_grpo.*`.
log "pip install -e . (flow_grpo package)..."
python3 -m pip install --quiet -e .

# --- HF auth + pre-cache SD3.5-M ---
log "HF login + pre-cache SD3.5-M..."
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
python3 <<'PY'
import time
from diffusers import StableDiffusion3Pipeline
import torch
t0 = time.time()
_ = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16,
)
print(f"  cached SD3.5-M in {time.time()-t0:.0f}s")
PY

# --- Append our compressibility_sd3_1gpu config to flow_grpo ---
# Inherits general_ocr_sd3_1gpu sizing (1xA100 40GB, LoRA, SD3.5-M, 512px,
# train_batch=8, num_image_per_prompt=8) and swaps the reward to in-process
# jpeg_compressibility. Prompts stay general_ocr (dataset/ocr/{train,test}.txt).
CONFIG_FILE="$FLOW_GRPO_DIR/config/grpo.py"
# Idempotent re-write: drop any prior compressibility_sd3_1gpu block, then
# append the current one. Lets us iterate on the config without re-cloning.
python3 - "$CONFIG_FILE" <<'STRIP_PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
# Strip any prior compressibility_sd3_1gpu function definition (up to next top-level def or EOF).
new = re.sub(
    r"\n\ndef compressibility_sd3_1gpu\([^)]*\):.*?(?=\n\ndef |\Z)",
    "",
    src,
    flags=re.DOTALL,
)
if new != src:
    p.write_text(new)
STRIP_PY

if ! grep -q "def compressibility_sd3_1gpu" "$CONFIG_FILE"; then
    log "Appending compressibility_sd3_1gpu config..."
    cat >> "$CONFIG_FILE" <<'PYCONFIG'


def compressibility_sd3_1gpu():
    config = general_ocr_sd3_1gpu()
    config.reward_fn = {"jpeg_compressibility": 1.0}
    config.save_dir = "logs/compressibility/sd3.5-M-1gpu-lora"
    # Mini run on 1xA100 40GB: original 1gpu config OOMs at first training
    # step (~39GB used). Halve both train_batch_size and num_image_per_prompt
    # (the sampler asserts batch_size is a multiple of num_image_per_prompt),
    # and disable EMA which keeps a full model shadow copy.
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 4
    config.train.batch_size = config.sample.train_batch_size
    config.sample.num_batches_per_epoch = int(8 / (1 * config.sample.train_batch_size / config.sample.num_image_per_prompt))
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.ema = False
    # Save every epoch and skip test-set eval (training reward is logged inline).
    config.save_freq = 1
    config.eval_freq = 999
    return config
PYCONFIG
fi

# Patch out the initial eval call. Without this, train_sd3.py runs eval at
# epoch 0 unconditionally (epoch 0 % 999 == 0), which costs ~37 min on this
# config and isn't needed for the training-reward curve.
TRAIN_PY="$FLOW_GRPO_DIR/scripts/train_sd3.py"
if ! grep -q "epoch > 0 and epoch % config.eval_freq" "$TRAIN_PY"; then
    log "Patching train_sd3.py to skip initial eval..."
    sed -i 's/if epoch % config.eval_freq == 0:/if epoch > 0 and epoch % config.eval_freq == 0:/' "$TRAIN_PY"
fi

# Force bf16 model load. Upstream script omits torch_dtype on from_pretrained,
# so SD3.5-M loads as float32 (~22GB transformer alone) and OOMs on 40GB. With
# bf16 the transformer is ~11GB and the run fits.
if ! grep -q "torch_dtype=torch.bfloat16" "$TRAIN_PY"; then
    log "Patching train_sd3.py to load SD3.5 as bfloat16..."
    sed -i 's|        config.pretrained.model$|        config.pretrained.model, torch_dtype=torch.bfloat16|' "$TRAIN_PY"
fi

# After bf16 load, LoRA params inherit bf16, but PyTorch GradScaler.unscale_()
# isn't implemented for bf16 — crashes mid-step-1 with
# `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16`.
# Cast just the trainable params back to fp32 (standard bf16-base/fp32-LoRA pattern).
if ! grep -q "LORA_FP32_CAST" "$TRAIN_PY"; then
    log "Patching train_sd3.py to keep LoRA params in fp32..."
    python3 - "$TRAIN_PY" <<'CAST_PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
needle = "    transformer = pipeline.transformer\n"
patch = (
    "    transformer = pipeline.transformer\n"
    "    # LORA_FP32_CAST: keep LoRA params in fp32 even when base is bf16,\n"
    "    # so the GradScaler unscale step has fp32 grads to operate on.\n"
    "    for _p in transformer.parameters():\n"
    "        if _p.requires_grad:\n"
    "            _p.data = _p.data.float()\n"
)
assert needle in src, "anchor missing — train_sd3.py changed"
src = src.replace(needle, patch, 1)
p.write_text(src)
CAST_PY
fi

# --- Train, capped at WALL_SECONDS ---
log "Launching FlowGRPO training (timeout ${WALL_SECONDS}s)..."
mkdir -p "$PROJECT_ROOT/outputs/checkpoints/flowgrpo"
mkdir -p "$PROJECT_ROOT/logs"

cd "$FLOW_GRPO_DIR"

# Note: timeout sends SIGTERM at deadline; train_sd3.py's `while True` loop
# will exit at the next yield. Last checkpoint at save_freq survives.
set +e
timeout --signal=TERM "${WALL_SECONDS}s" \
    accelerate launch \
        --config_file scripts/accelerate_configs/multi_gpu.yaml \
        --num_processes=1 \
        --main_process_port 29501 \
        scripts/train_sd3.py \
        --config config/grpo.py:compressibility_sd3_1gpu \
    2>&1 | tee "$PROJECT_ROOT/logs/flowgrpo_train.log"
RC=$?
set -e

if [ $RC -eq 124 ]; then
    log "Wall-clock budget reached ($WALL_HOURS h). Training stopped on schedule."
elif [ $RC -ne 0 ]; then
    log "WARN: training exited rc=$RC. Check logs/flowgrpo_train.log."
fi

# --- Pull the latest checkpoint into our outputs tree ---
LATEST_CKPT=$(ls -td "$FLOW_GRPO_DIR"/logs/compressibility/sd3.5-M-1gpu-lora/*/checkpoints/checkpoint_* 2>/dev/null | head -1 || true)
if [ -n "$LATEST_CKPT" ]; then
    log "Latest checkpoint: $LATEST_CKPT"
    cp -r "$LATEST_CKPT" "$PROJECT_ROOT/outputs/checkpoints/flowgrpo/" || true
fi

log "=== FlowGRPO mini run done ==="
log "Logs: logs/flowgrpo_train.log"
log "Checkpoints: outputs/checkpoints/flowgrpo/"
log "Next: rsync results back, then bash scripts/cloud/terminate.sh"
