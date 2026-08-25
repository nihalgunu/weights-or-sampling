#!/bin/bash
# Runs ON the Lambda instance. Scores LoRA checkpoints from the FlowGRPO+CLIP+OCR
# run against CLIP reward, on flow_grpo's held-out OCR test set.
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_flowgrpo_clip_ocr_eval.sh

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ -z "${HF_TOKEN:-}" ]; then
    log "ERROR: HF_TOKEN not set."
    exit 1
fi

log "=== FlowGRPO+CLIP+OCR LoRA eval starting ==="

python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet --upgrade --force-reinstall pillow
python3 -m pip install --quiet \
    'diffusers>=0.31,<0.40' \
    'transformers==4.49.0' \
    'peft>=0.12' \
    sentencepiece protobuf safetensors

python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"

# Use flow_grpo's held-out OCR test set for eval. 30 prompts, paired seeds.
PROMPTS_FILE="$PROJECT_ROOT/repos/flow_grpo/dataset/ocr/test.txt"
if [ ! -f "$PROMPTS_FILE" ]; then
    log "ERROR: $PROMPTS_FILE missing — flow_grpo not cloned?"
    exit 1
fi

# Pick endpoints: no_lora baseline + early sanity + ~300-step (DDRL-matched) + final.
ENDPOINTS=$(python3 - "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip_ocr" <<'PY'
import os, sys, re, pathlib
root = pathlib.Path(sys.argv[1])
ckpts = sorted(
    [d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("checkpoint")],
    key=lambda n: int(re.search(r"\d+", n).group()),
)
if not ckpts:
    print("no_lora"); sys.exit(0)
nums = [int(re.search(r"\d+", n).group()) for n in ckpts]
target = 300
mid_idx = min(range(len(nums)), key=lambda i: abs(nums[i] - target))
picks = []
for i in (0, mid_idx, len(ckpts) - 1):
    if ckpts[i] not in picks:
        picks.append(ckpts[i])
print(" ".join(["no_lora"] + picks))
PY
)
log "Endpoints: $ENDPOINTS"

mkdir -p "$PROJECT_ROOT/outputs"
python3 "$PROJECT_ROOT/scripts/cloud/eval_lora_compressibility.py" \
    --base stabilityai/stable-diffusion-3.5-medium \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip_ocr" \
    --prompts-file "$PROMPTS_FILE" \
    --num-prompts 30 \
    --num-steps 40 \
    --reward clip \
    --endpoints $ENDPOINTS \
    --out "$PROJECT_ROOT/outputs/flowgrpo_clip_ocr_eval.json"

log "=== FlowGRPO+CLIP+OCR LoRA eval done ==="
log "Result: outputs/flowgrpo_clip_ocr_eval.json"
