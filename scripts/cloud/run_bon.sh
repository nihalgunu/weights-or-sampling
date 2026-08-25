#!/bin/bash
# Runs ON the box. Best-of-N (N=4) matched-inference-compute baseline for the
# clip and pickscore settings. Expects uploaded LoRAs:
#   outputs/checkpoints/flowgrpo_clip_pub_s42/checkpoint-262/lora
#   outputs/checkpoints/flowgrpo_pickscore_pub_s42/checkpoint-228/lora
# Usage: HF_TOKEN=... bash scripts/cloud/run_bon.sh

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

CLIP_LORA=$(ls -d "$PROJECT_ROOT"/outputs/checkpoints/flowgrpo_clip_pub_s42/checkpoint-*/lora 2>/dev/null | head -1)
PS_LORA=$(ls -d "$PROJECT_ROOT"/outputs/checkpoints/flowgrpo_pickscore_pub_s42/checkpoint-*/lora 2>/dev/null | head -1)
[ -n "$CLIP_LORA" ] || { log "FATAL: clip LoRA missing"; exit 1; }
[ -n "$PS_LORA" ] || { log "FATAL: pickscore LoRA missing"; exit 1; }

log ">>> BoN clip setting"
python3 "$PROJECT_ROOT/scripts/cloud/eval_bon.py" --reward clip \
    --prompts-file "$PROMPTS_FILE" --num-prompts 150 --n 4 \
    --seed-bases 0,1000,2000 \
    --rl-lora "rl_s42=$CLIP_LORA" \
    --out "$PROJECT_ROOT/outputs/study_bon_clip_matrix.json" \
    || { log "FATAL: bon clip failed"; exit 1; }

log ">>> BoN pickscore setting"
python3 "$PROJECT_ROOT/scripts/cloud/eval_bon.py" --reward pickscore \
    --prompts-file "$PROMPTS_FILE" --num-prompts 150 --n 4 \
    --seed-bases 0,1000,2000 \
    --rl-lora "rl_s42=$PS_LORA" \
    --out "$PROJECT_ROOT/outputs/study_bon_pickscore_matrix.json" \
    || { log "FATAL: bon pickscore failed"; exit 1; }

log "=== BON done ==="
