#!/bin/bash
# Runs ON the box. Two conditions-map evals (eval-only):
#  1. COMPHARD: compositional hard prompts (counting/relations, detector base ~0.5
#     expected) — does either optimizer convert CLIP gains into correctness when
#     there IS headroom? arms: base, rho20, rho20+mu20, RL(clip s42).
#  2. SHIFT: cross-distribution transfer — CLIP-trained RL LoRA and TFG evaluated
#     on the OCR prompt distribution (and OCR-trained RL on GenEval prompts):
#     does a banked gain transfer across domains vs per-sample guidance?
# Usage: HF_TOKEN=... bash scripts/cloud/run_cmap.sh
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
python3 -m pip install --quiet --force-reinstall 'numpy==1.26.4'
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
git clone -q https://github.com/yifan123/flow_grpo "$PROJECT_ROOT/repos/flow_grpo" 2>/dev/null || true

# generate compositional hard prompts (parser-compatible templates)
CH="$PROJECT_ROOT/outputs/comphard_prompts.txt"
python3 - "$CH" <<'PY'
import random, sys, pathlib
random.seed(7)
objs = ["cat","dog","apple","chair","book","bird","cup","car","bottle","clock","vase","shoe","hat","ball","fork"]
colors = ["red","blue","green","yellow","purple","orange","black","white"]
nums = ["two","three","four","five"]
rels = ["above","below","to the left of","to the right of"]
out = []
while len(out) < 60:
    p = f"a photo of {random.choice(nums)} {random.choice(objs)}s"
    if p not in out: out.append(p)
while len(out) < 110:
    a, b = random.sample(objs, 2)
    p = f"a photo of a {random.choice(colors)} {a} and a {random.choice(colors)} {b}"
    if p not in out: out.append(p)
while len(out) < 150:
    a, b = random.sample(objs, 2)
    p = f"a photo of a {a} {random.choice(rels)} a {b}"
    if p not in out: out.append(p)
pathlib.Path(sys.argv[1]).write_text("\n".join(out) + "\n")
print(len(out), "comphard prompts")
PY

CLIP_LORA=$(ls -d "$PROJECT_ROOT"/outputs/checkpoints/flowgrpo_clip_pub_s42/checkpoint-*/lora 2>/dev/null | head -1)
OCR_LORA=$(ls -d "$PROJECT_ROOT"/outputs/checkpoints/flowgrpo_clip_ocr_pub_s42/checkpoint-*/lora 2>/dev/null | head -1)
[ -n "$CLIP_LORA" ] && [ -n "$OCR_LORA" ] || { log "FATAL: LoRAs missing"; exit 1; }

log ">>> COMPHARD eval"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward clip --prompts-file "$CH" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos 20 --tfg-combos 20:20 --stack-rhos "" \
    --rl-lora "rl_clip_s42=$CLIP_LORA" --seed-bases 0,1000 \
    --image-dir "$PROJECT_ROOT/outputs/study_comphard_images" \
    --out "$PROJECT_ROOT/outputs/study_comphard_matrix.json" \
    || { log "FATAL: comphard eval failed"; exit 1; }
log ">>> COMPHARD detector"
python3 "$PROJECT_ROOT/scripts/cloud/detector_score.py" \
    --image-root "$PROJECT_ROOT/outputs/study_comphard_images" \
    --prompts-file "$CH" --num-prompts 150 \
    --out "$PROJECT_ROOT/outputs/study_comphard_detector.json" || log "WARN detector"

log ">>> SHIFT eval A (clip-RL + TFG on OCR distribution)"
OCRP="$PROJECT_ROOT/repos/flow_grpo/dataset/ocr/test.txt"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward clip --prompts-file "$OCRP" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos 20 --stack-rhos "" \
    --rl-lora "rl_clip_s42=$CLIP_LORA" --seed-bases 0,1000 \
    --out "$PROJECT_ROOT/outputs/study_shift_clip2ocr_matrix.json" \
    || { log "FATAL: shift A failed"; exit 1; }
log ">>> SHIFT eval B (ocr-RL on GenEval distribution)"
GP="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$GP" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward clip --prompts-file "$GP" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos "" --stack-rhos "" \
    --rl-lora "rl_ocr_s42=$OCR_LORA" --seed-bases 0,1000 \
    --out "$PROJECT_ROOT/outputs/study_shift_ocr2gen_matrix.json" \
    || { log "FATAL: shift B failed"; exit 1; }
log "=== CMAP done ==="
