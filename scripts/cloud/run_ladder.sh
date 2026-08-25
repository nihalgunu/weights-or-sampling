#!/bin/bash
# Runs ON the box. Dissociation strength-ladder on the OCR setting:
# arms = base, rho{5,10,20,40}, mu20, rho20+mu20, RL(b=0.04 s42 LoRA uploaded);
# images saved; task scoring with EasyOCR AND tesseract (word-F1 + similarity).
# Usage: HF_TOKEN=... bash scripts/cloud/run_ladder.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?}"
log ">>> deps"
sudo apt-get install -y -qq tesseract-ocr >/dev/null 2>&1 || log "WARN: tesseract apt"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet --upgrade --force-reinstall pillow
python3 -m pip install --quiet 'diffusers>=0.31,<0.40' 'transformers==4.49.0' \
    'peft>=0.12' 'accelerate>=0.33' sentencepiece protobuf safetensors torchvision tqdm easyocr pytesseract
python3 -m pip install --quiet --force-reinstall 'numpy==1.26.4'
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
# OCR prompts: use up to 300 from test.txt
git clone -q https://github.com/yifan123/flow_grpo "$PROJECT_ROOT/repos/flow_grpo" 2>/dev/null || true
OCR_FILE="$PROJECT_ROOT/repos/flow_grpo/dataset/ocr/test.txt"
NP=$(grep -c . "$OCR_FILE"); NP=$(( NP < 300 ? NP : 300 ))
log "prompts: $NP"
L=$(ls -d "$PROJECT_ROOT"/outputs/checkpoints/flowgrpo_clip_ocr_pub_s42/checkpoint-*/lora 2>/dev/null | head -1)
RL_ARGS=(); [ -n "$L" ] && RL_ARGS+=(--rl-lora "rl_s42=$L")
log ">>> ladder eval (images saved)"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward clip --prompts-file "$OCR_FILE" --num-prompts "$NP" --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos 5,10,20,40 --tfg-mus 20 --tfg-combos 20:20 --stack-rhos "" "${RL_ARGS[@]}" \
    --seed-bases 0,1000 \
    --image-dir "$PROJECT_ROOT/outputs/study_ladder_images" \
    --out "$PROJECT_ROOT/outputs/study_ladder_matrix.json" \
    || { log "FATAL: ladder eval failed"; exit 1; }
log ">>> dual-engine OCR task scoring"
python3 - "$OCR_FILE" "$NP" <<'PY'
import json, pathlib, re, sys
from difflib import SequenceMatcher
import easyocr, pytesseract
from PIL import Image
prompts = [l.strip() for l in open(sys.argv[1]) if l.strip()][:int(sys.argv[2])]
targets = []
for p in prompts:
    m = re.findall(r'["“]([^"”]+)["”]', p)
    targets.append(m[0].strip().lower() if m else None)
reader = easyocr.Reader(["en"], gpu=True, verbose=False)
def wf1(tgt, det):
    tw, dw = set(tgt.split()), set(det.split())
    if not tw: return None
    if not dw: return 0.0
    inter = len(tw & dw)
    p_ = inter/len(dw); r_ = inter/len(tw)
    return 0.0 if p_+r_ == 0 else 2*p_*r_/(p_+r_)
root = pathlib.Path("outputs/study_ladder_images")
out = {}
for arm_dir in sorted(root.iterdir()):
    if not arm_dir.is_dir(): continue
    arm = {}
    for b_dir in sorted(arm_dir.iterdir()):
        rows = {"easy_sim": [], "easy_f1": [], "tess_sim": [], "tess_f1": []}
        for i, tgt in enumerate(targets):
            p = b_dir / f"{i:03d}.png"
            if tgt is None or not p.exists():
                for k in rows: rows[k].append(None)
                continue
            try: e = " ".join(reader.readtext(str(p), detail=0)).lower()
            except Exception: e = ""
            try: t = pytesseract.image_to_string(Image.open(p)).lower()
            except Exception: t = ""
            rows["easy_sim"].append(SequenceMatcher(None, tgt, e).ratio())
            rows["easy_f1"].append(wf1(tgt, e))
            rows["tess_sim"].append(SequenceMatcher(None, tgt, t).ratio())
            rows["tess_f1"].append(wf1(tgt, t))
        arm[b_dir.name] = rows
    out[arm_dir.name] = arm
    print(f"[ocr] {arm_dir.name} scored", flush=True)
    pathlib.Path("outputs/study_ladder_taskocr.json").write_text(json.dumps(out))
PY
log "=== LADDER done ==="
