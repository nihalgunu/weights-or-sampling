#!/bin/bash
# Runs ON the Lambda instance. Scores LoRA checkpoints from the FlowGRPO+CLIP
# run against CLIP reward, on the same prompt set DDRL used.
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_flowgrpo_clip_eval.sh

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

log "=== FlowGRPO+CLIP LoRA eval starting ==="

# Deps already installed by run_flowgrpo_clip.sh; re-asserting for safety in
# case eval runs on a fresh box.
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet --upgrade --force-reinstall pillow
python3 -m pip install --quiet \
    'diffusers>=0.31,<0.40' \
    'transformers==4.49.0' \
    'peft>=0.12' \
    sentencepiece protobuf safetensors

python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"

# Write the DDRL prompt set to a flat .txt that the eval script can read,
# in the same order as DDRL's prompts_smoke.json (paired-seed comparison).
PROMPTS_TXT="$PROJECT_ROOT/outputs/eval_prompts_ddrl.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" "$PROMPTS_TXT" <<'PY'
import json, sys, pathlib
src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
prompts = [d["prompt"] for d in json.loads(src.read_text())]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(prompts) + "\n")
print(f"wrote {len(prompts)} prompts → {out}")
PY

# Pick the endpoints: no_lora baseline + early + matched-DDRL-compute (300
# grad updates ≈ a checkpoint near the middle of the run) + final. We don't
# know the exact step numbers ahead of time, so list every checkpoint dir and
# pick endpoints at the boundaries + middle.
ENDPOINTS=$(python3 - "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip" <<'PY'
import os, sys, re, pathlib
root = pathlib.Path(sys.argv[1])
ckpts = sorted(
    [d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("checkpoint")],
    key=lambda n: int(re.search(r"\d+", n).group()),
)
if not ckpts:
    print("no_lora")
    sys.exit(0)
nums = [int(re.search(r"\d+", n).group()) for n in ckpts]
# Pick: first, one closest to 300 (matched-DDRL-compute), and last.
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
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip" \
    --prompts-file "$PROMPTS_TXT" \
    --num-prompts 30 \
    --num-steps 40 \
    --reward clip \
    --endpoints $ENDPOINTS \
    --out "$PROJECT_ROOT/outputs/flowgrpo_clip_eval.json"

log "=== FlowGRPO+CLIP LoRA eval done ==="
log "Result: outputs/flowgrpo_clip_eval.json"
