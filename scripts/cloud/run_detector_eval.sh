#!/bin/bash
# Runs ON the box. Detector-based (GenEval-style) scoring pass:
#   1. Regenerate the CLIP-setting images (lost with the original box) from
#      the pulled LoRAs: base + TFG rho20 + RL s42/s2042 + stacked s42.
#   2. SigLIP external metrics on the regenerated CLIP tree (backfills the
#      other lost artifact).
#   3. OWLv2+CLIP detector scoring (presence/count/color/position) on BOTH
#      the regenerated CLIP tree and the uploaded PickScore-setting tree.
#
# Expects (uploaded by the driver before kick):
#   outputs/study_pickscore_pub_images.tar.gz
#   outputs/checkpoints/flowgrpo_clip_pub_s42/checkpoint-262/lora
#   outputs/checkpoints/flowgrpo_clip_pub_s2042/checkpoint-264/lora
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_detector_eval.sh

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?HF_TOKEN required}"

log "=== DETEVAL start ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

log ">>> deps"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet --upgrade --force-reinstall pillow
python3 -m pip install --quiet 'diffusers>=0.31,<0.40' 'transformers==4.49.0' \
    'peft>=0.12' 'accelerate>=0.33' sentencepiece protobuf safetensors torchvision tqdm scipy
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"

# --- prompts (identical construction to the pub run) ---
PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" \
          "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$PROMPTS_FILE" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
out = smoke + [p for p in train if p not in set(smoke)][:120]
pathlib.Path(sys.argv[3]).write_text("\n".join(out) + "\n")
print(f"wrote {len(out)} prompts")
PY

# --- untar pickscore images ---
if [ ! -d "$PROJECT_ROOT/outputs/study_pickscore_pub_images" ]; then
    log ">>> untar pickscore images"
    tar xzf "$PROJECT_ROOT/outputs/study_pickscore_pub_images.tar.gz" -C "$PROJECT_ROOT/outputs" \
        || { log "FATAL: untar failed"; exit 1; }
fi

# --- 1. regenerate CLIP-setting images ---
S42="$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip_pub_s42/checkpoint-262/lora"
S2042="$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip_pub_s2042/checkpoint-264/lora"
[ -d "$S42" ] || { log "FATAL: $S42 missing"; exit 1; }
[ -d "$S2042" ] || { log "FATAL: $S2042 missing"; exit 1; }
if [ ! -f "$PROJECT_ROOT/outputs/study_clip_pub_matrix_regen.json" ]; then
    log ">>> regenerating CLIP-setting images (5 arms x 450)"
    python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
        --reward clip \
        --prompts-file "$PROMPTS_FILE" \
        --num-prompts 150 --num-steps 40 \
        --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
        --tfg-rhos 20 --stack-rhos "" \
        --rl-lora "rl_s42=$S42" --rl-lora "rl_s2042=$S2042" \
        --stack "stack_s42_rho20=$S42:20" \
        --seed-bases 0,1000,2000 \
        --image-dir "$PROJECT_ROOT/outputs/study_clip_pub_images" \
        --out "$PROJECT_ROOT/outputs/study_clip_pub_matrix_regen.json" \
        || { log "FATAL: clip regen failed"; exit 1; }
fi

# --- 2. SigLIP on the regenerated CLIP tree ---
log ">>> SigLIP external metrics (clip tree)"
python3 "$PROJECT_ROOT/scripts/cloud/external_metric.py" \
    --image-root "$PROJECT_ROOT/outputs/study_clip_pub_images" \
    --prompts-file "$PROMPTS_FILE" --num-prompts 150 \
    --metrics siglip \
    --out "$PROJECT_ROOT/outputs/study_clip_pub_external.json" \
    || log "WARN: clip siglip failed"

# --- 3. detector scoring on both trees ---
log ">>> detector scoring (clip tree)"
python3 "$PROJECT_ROOT/scripts/cloud/detector_score.py" \
    --image-root "$PROJECT_ROOT/outputs/study_clip_pub_images" \
    --prompts-file "$PROMPTS_FILE" --num-prompts 150 \
    --out "$PROJECT_ROOT/outputs/study_clip_pub_detector.json" \
    || { log "FATAL: clip detector scoring failed"; exit 1; }

log ">>> detector scoring (pickscore tree)"
python3 "$PROJECT_ROOT/scripts/cloud/detector_score.py" \
    --image-root "$PROJECT_ROOT/outputs/study_pickscore_pub_images" \
    --prompts-file "$PROMPTS_FILE" --num-prompts 150 \
    --out "$PROJECT_ROOT/outputs/study_pickscore_pub_detector.json" \
    || { log "FATAL: pickscore detector scoring failed"; exit 1; }

log "=== DETEVAL done ==="
