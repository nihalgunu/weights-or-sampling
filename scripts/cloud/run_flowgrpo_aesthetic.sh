#!/bin/bash
# FlowGRPO+PickScore on the same DDRL prompt set.
#
# Why this run: two FlowGRPO+CLIP runs (DDRL prompts, OCR prompts) both
# stalled or regressed at our 1×A100/batch=2/~600-step regime. Tests whether
# CLIP-L is the wrong reward signal at small scale (it rewards prompt
# complexity more than image fidelity) or whether small-scale RL on
# diffusion is broadly stuck. PickScore is a human-preference model
# (yuvalkirstain/PickScore_v1) — denser per-sample signal, native to
# flow_grpo (rewards.py `pickscore`). Same DDRL prompts → directly
# comparable to DDRL's CLIP 0% and the prior FlowGRPO+CLIP+DDRL +0.015.
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_flowgrpo_aesthetic.sh [WALL_HOURS]

set -euo pipefail

WALL_HOURS="${1:-4}"
WALL_SECONDS=$(python3 -c "print(int(float('$WALL_HOURS')*3600))")

export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ -z "${HF_TOKEN:-}" ]; then
    log "ERROR: HF_TOKEN not set. SD3.5-M is gated."
    exit 1
fi

log "=== FlowGRPO+PickScore run starting (budget: ${WALL_HOURS}h / ${WALL_SECONDS}s) ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

FLOW_GRPO_COMMIT="main"
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
    sentencepiece protobuf safetensors pyyaml tqdm huggingface_hub

# ARM (GH200): the pinned bitsandbytes==0.45.3 has no aarch64 wheel and we
# don't use it in the LoRA path — relax the pin before installing.
sed -i.bak 's/bitsandbytes==[0-9.]*/bitsandbytes>=0.42.0/' setup.py pyproject.toml requirements.txt 2>/dev/null || true
sed -i.bak2 '/xformers/d' setup.py 2>/dev/null || true
log "pip install -e . (flow_grpo package)..."
python3 -m pip install --quiet -e .

log "HF login + pre-cache SD3.5-M + PickScore (CLIP-H + PickScore_v1)..."
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
python3 <<'PY'
import time
from diffusers import StableDiffusion3Pipeline
from transformers import CLIPModel, CLIPProcessor
import torch
t0 = time.time()
import os
_ = StableDiffusion3Pipeline.from_pretrained(
    os.environ.get("FG_MODEL", "stabilityai/stable-diffusion-3.5-medium"), torch_dtype=torch.bfloat16,
)
print(f"  cached SD3.5-M in {time.time()-t0:.0f}s")
t0 = time.time()
_ = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
_ = CLIPModel.from_pretrained("yuvalkirstain/PickScore_v1")
print(f"  cached PickScore_v1 in {time.time()-t0:.0f}s")
PY

# --- Write DDRL prompt set into a flow_grpo dataset dir (apples-to-apples) ---
PROMPTS_JSON="$PROJECT_ROOT/scripts/cloud/prompts_smoke.json"
[ -f "$PROMPTS_JSON" ] || { log "ERROR: $PROMPTS_JSON missing"; exit 1; }
DATASET_DIR="$FLOW_GRPO_DIR/dataset/ddrl_aesthetic"
mkdir -p "$DATASET_DIR"
python3 - "$PROMPTS_JSON" "$DATASET_DIR" <<'PROMPTS_PY'
import json, sys, pathlib
src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
prompts = [d["prompt"] for d in json.loads(src.read_text())]
(out / "train.txt").write_text("\n".join(prompts) + "\n")
(out / "test.txt").write_text("\n".join(prompts) + "\n")
print(f"wrote {len(prompts)} prompts → {out}/{{train,test}}.txt")
PROMPTS_PY

# --- Append aesthetic_sd3_1gpu config to flow_grpo ---
CONFIG_FILE="$FLOW_GRPO_DIR/config/grpo.py"
python3 - "$CONFIG_FILE" <<'STRIP_PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1]); src = p.read_text()
new = re.sub(
    r"\n\ndef aesthetic_sd3_1gpu\([^)]*\):.*?(?=\n\ndef |\Z)",
    "", src, flags=re.DOTALL,
)
if new != src: p.write_text(new)
STRIP_PY

if ! grep -q "def aesthetic_sd3_1gpu" "$CONFIG_FILE"; then
    log "Appending aesthetic_sd3_1gpu config..."
    cat >> "$CONFIG_FILE" <<PYCONFIG


def aesthetic_sd3_1gpu():
    config = general_ocr_sd3_1gpu()
    # Reward = PickScore_v1 (human-preference model, dense per-sample signal).
    config.reward_fn = {"aesthetic": 1.0}
    # Same DDRL prompt set written by runner — apples-to-apples with DDRL CLIP 0%
    # and the prior FlowGRPO+CLIP+DDRL Δ=+0.015.
    config.dataset = os.path.join(os.getcwd(), "dataset/ddrl_aesthetic")
    config.save_dir = "logs/aesthetic/sd3.5-M-1gpu-lora"
    # PickScore (ViT-H/14) is ~2× the size of CLIP-L; same batch budget but
    # tighter. CLIP-L+batch=2 sat at 32.7GB on a 40GB A100 last run, leaving
    # ~7GB headroom — PickScore should fit at the same batch.
    # GRPO group size = num_image_per_prompt. At group=2 the within-group
    # advantage is degenerate (one win, one loss -> +/-1): per-step variance
    # swamps the mean, which is why PickScore/CLIP regressed at our 1xA100 scale.
    # FG_GROUP raises the rollout-per-prompt for real variance reduction. On 1 GPU
    # the sampler requires train_batch_size % num_image_per_prompt == 0, so
    # train_batch_size == group (one prompt, group rollouts per sampling iter).
    # group=8 matches upstream general_ocr_sd3_1gpu and needs an 80GB GPU (group=2
    # was only to fit 40GB). FG_NBPE = sampling iters (prompts) per epoch;
    # grad-accum = FG_NBPE//2 keeps two optimizer steps/epoch, each averaging
    # FG_NBPE prompts' gradients -- the gradient-accumulation lever Haotian asked
    # for. Defaults (group=2, nbpe=8) reproduce the prior 40GB runs exactly.
    _group = int(os.environ.get("FG_GROUP", "2"))
    _nbpe = int(os.environ.get("FG_NBPE", "8"))
    assert _nbpe % 2 == 0, "FG_NBPE must be even (grad-accum = FG_NBPE // 2)"
    config.sample.train_batch_size = _group
    config.sample.num_image_per_prompt = _group
    config.train.batch_size = _group
    config.sample.num_batches_per_epoch = _nbpe
    config.train.gradient_accumulation_steps = _nbpe // 2
    config.train.ema = False
    # FG_SEED: training seed for multi-seed replication (pub-grade study).
    config.seed = int(os.environ.get("FG_SEED", "42"))
    # FG_MODEL: base model override (SD3.5-Large replication).
    config.pretrained.model = os.environ.get("FG_MODEL", config.pretrained.model)
    # FG_BETA: policy-KL coefficient override (upstream default 0.04).
    config.train.beta = float(os.environ.get("FG_BETA", str(config.train.beta)))
    config.save_freq = 1
    config.eval_freq = 999
    return config
PYCONFIG
fi

# --- train_sd3.py patches (same as prior runners) ---
TRAIN_PY="$FLOW_GRPO_DIR/scripts/train_sd3.py"
if ! grep -q "epoch > 0 and epoch % config.eval_freq" "$TRAIN_PY"; then
    log "Patching train_sd3.py to skip initial eval..."
    sed -i 's/if epoch % config.eval_freq == 0:/if epoch > 0 and epoch % config.eval_freq == 0:/' "$TRAIN_PY"
fi

if ! grep -q "torch_dtype=torch.bfloat16" "$TRAIN_PY"; then
    log "Patching train_sd3.py to load SD3.5 as bfloat16..."
    sed -i 's|        config.pretrained.model$|        config.pretrained.model, torch_dtype=torch.bfloat16|' "$TRAIN_PY"
fi

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
log "Launching FlowGRPO+PickScore training (timeout ${WALL_SECONDS}s)..."
mkdir -p "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_aesthetic"
mkdir -p "$PROJECT_ROOT/logs"

cd "$FLOW_GRPO_DIR"
set +e
timeout --signal=TERM "${WALL_SECONDS}s" \
    accelerate launch \
        --config_file scripts/accelerate_configs/multi_gpu.yaml \
        --num_processes=1 \
        --main_process_port 29501 \
        scripts/train_sd3.py \
        --config config/grpo.py:aesthetic_sd3_1gpu \
    2>&1 | tee "$PROJECT_ROOT/logs/flowgrpo_aesthetic_train.log"
RC=$?
set -e

if [ $RC -eq 124 ]; then
    log "Wall-clock budget reached ($WALL_HOURS h). Training stopped on schedule."
elif [ $RC -ne 0 ]; then
    log "WARN: training exited rc=$RC. Check logs/flowgrpo_aesthetic_train.log."
fi

CKPT_DIR="$FLOW_GRPO_DIR/logs/aesthetic/sd3.5-M-1gpu-lora/checkpoints"
if [ -d "$CKPT_DIR" ]; then
    N=$(ls "$CKPT_DIR" | grep -c '^checkpoint-' || true)
    log "Run dir: $CKPT_DIR ($N checkpoints)"
    cp -r "$CKPT_DIR"/checkpoint-* "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_aesthetic/" 2>/dev/null || true
    log "copied $(ls $PROJECT_ROOT/outputs/checkpoints/flowgrpo_aesthetic/ | wc -l) checkpoints to outputs/"
else
    log "WARN: $CKPT_DIR missing — no checkpoints to copy"
fi

log "=== FlowGRPO+PickScore run done ==="
log "Logs: logs/flowgrpo_aesthetic_train.log"
log "Checkpoints: outputs/checkpoints/flowgrpo_aesthetic/"
