#!/bin/bash
# Runs ON the box. "Completion pack" (eval-only):
#   1. clip: best-known-recipe stacked arm (RL s42 + rho20 + mu20)
#   2. ocr: mu20 and rho20+mu20 arms (does the mu doubling generalize?)
#   3. pickscore: mu80, rho80+mu80, rho {640,1280} ceiling hunt, best-recipe stack
#   4. BoN N=16 on clip base (selection-scaling curve; per-candidate scores stored)
# Expects uploaded LoRAs: flowgrpo_clip_pub_s42, flowgrpo_pickscore_pub_s42.
# Usage: HF_TOKEN=... bash scripts/cloud/run_pack.sh

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

OCR_FILE="$PROJECT_ROOT/outputs/ocr_prompts.txt"
curl -sL https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/ocr/test.txt > "$OCR_FILE"
[ "$(grep -c . "$OCR_FILE")" -ge 150 ] || { log "FATAL: OCR prompt fetch failed"; exit 1; }

CLIP_LORA=$(ls -d "$PROJECT_ROOT"/outputs/checkpoints/flowgrpo_clip_pub_s42/checkpoint-*/lora 2>/dev/null | head -1)
PS_LORA=$(ls -d "$PROJECT_ROOT"/outputs/checkpoints/flowgrpo_pickscore_pub_s42/checkpoint-*/lora 2>/dev/null | head -1)
[ -n "$CLIP_LORA" ] && [ -n "$PS_LORA" ] || { log "FATAL: LoRAs missing"; exit 1; }

log ">>> [1/4] clip best-recipe stack"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward clip --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos "" --stack-rhos "" \
    --rl-lora "rl_s42=$CLIP_LORA" \
    --stack "stackfull_s42=$CLIP_LORA:20:20" \
    --seed-bases 0,1000,2000 \
    --out "$PROJECT_ROOT/outputs/study_pack_clip_matrix.json" \
    || { log "FATAL: pack clip failed"; exit 1; }

log ">>> [2/4] ocr mu arms"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward clip --prompts-file "$OCR_FILE" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/empty_ckpts_x" \
    --tfg-rhos "" --tfg-mus 20 --tfg-combos 20:20 --stack-rhos "" \
    --seed-bases 0,1000,2000 \
    --out "$PROJECT_ROOT/outputs/study_pack_ocr_matrix.json" \
    || { log "FATAL: pack ocr failed"; exit 1; }

log ">>> [3/4] pickscore mu + ceiling + best-recipe"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward pickscore --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos 640,1280 --tfg-mus 80 --tfg-combos 80:80 --stack-rhos "" \
    --rl-lora "rl_s42=$PS_LORA" \
    --stack "stackfull_s42=$PS_LORA:80:80" \
    --seed-bases 0,1000,2000 \
    --out "$PROJECT_ROOT/outputs/study_pack_pickscore_matrix.json" \
    || { log "FATAL: pack pickscore failed"; exit 1; }

log ">>> [4/4] BoN N=16 selection curve (clip, 2 seed bases)"
python3 "$PROJECT_ROOT/scripts/cloud/eval_bon.py" --reward clip \
    --prompts-file "$PROMPTS_FILE" --num-prompts 150 --n 16 \
    --seed-bases 0,1000 \
    --out "$PROJECT_ROOT/outputs/study_bon16_clip_matrix.json" \
    || { log "FATAL: bon16 failed"; exit 1; }

log "=== PACK done ==="
