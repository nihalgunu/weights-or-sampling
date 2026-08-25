#!/bin/bash
# Runs ON the box. Full GenEval prompt set (553) for the headline arms of the
# clip setting, single eval seed base, + detector scoring.
# Expects uploaded LoRA: outputs/checkpoints/flowgrpo_clip_pub_s42/checkpoint-*/lora
# Usage: HF_TOKEN=... bash scripts/cloud/run_g553.sh

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
    'peft>=0.12' 'accelerate>=0.33' sentencepiece protobuf safetensors torchvision tqdm scipy
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"

log ">>> fetch GenEval evaluation prompts (553)"
G553="$PROJECT_ROOT/outputs/g553_prompts.txt"
curl -sL https://raw.githubusercontent.com/djghosh13/geneval/main/prompts/evaluation_metadata.jsonl \
  | python3 -c "
import json, sys
lines = [json.loads(l)['prompt'] for l in sys.stdin if l.strip()]
print('\n'.join(lines))
" > "$G553"
NP=$(grep -c . "$G553")
log "prompts: $NP"
[ "$NP" -ge 500 ] || { log "FATAL: prompt fetch failed ($NP)"; exit 1; }

CLIP_LORA=$(ls -d "$PROJECT_ROOT"/outputs/checkpoints/flowgrpo_clip_pub_s42/checkpoint-*/lora 2>/dev/null | head -1)
[ -n "$CLIP_LORA" ] || { log "FATAL: clip LoRA missing"; exit 1; }

log ">>> G553 eval matrix"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward clip --prompts-file "$G553" --num-prompts "$NP" --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos 20 --tfg-combos 20:20 --stack-rhos "" \
    --rl-lora "rl_s42=$CLIP_LORA" \
    --stack "stack_s42_rho20=$CLIP_LORA:20" \
    --seed-bases 0 \
    --image-dir "$PROJECT_ROOT/outputs/study_g553_images" \
    --out "$PROJECT_ROOT/outputs/study_g553_matrix.json" \
    || { log "FATAL: eval failed"; exit 1; }

log ">>> detector scoring"
python3 "$PROJECT_ROOT/scripts/cloud/detector_score.py" \
    --image-root "$PROJECT_ROOT/outputs/study_g553_images" \
    --prompts-file "$G553" --num-prompts "$NP" \
    --out "$PROJECT_ROOT/outputs/study_g553_detector.json" \
    || log "WARN: detector scoring failed"

tar czf "$PROJECT_ROOT/outputs/study_g553_images.tar.gz" -C "$PROJECT_ROOT/outputs" study_g553_images || true
log "=== G553 done ==="
