#!/bin/bash
# Runs ON the box. (a) ImageReward (BLIP-family) external scoring over the
# saved OCR / PickScore / ensemble / G553 image trees (uploaded as tars);
# (b) regenerate PickScore high-rho images (320/640/1280) and audit them with
# the detector + ImageReward, per the over-optimization standard.
# Usage: HF_TOKEN=... bash scripts/cloud/run_probe2.sh
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
    'peft>=0.12' 'accelerate>=0.33' sentencepiece protobuf safetensors torchvision tqdm image-reward
python3 -m pip install --quiet --force-reinstall 'numpy==1.26.4' 'scipy==1.11.4'
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$PROMPTS_FILE" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY
for T in study_pickscore_pub_images study_clip_ocr_pub_images study_ensemble_images; do
  [ -f "$PROJECT_ROOT/outputs/$T.tar.gz" ] || mv "$PROJECT_ROOT/outputs/checkpoints/$T.tar.gz" "$PROJECT_ROOT/outputs/" 2>/dev/null || true
  [ -d "$PROJECT_ROOT/outputs/$T" ] || tar xzf "$PROJECT_ROOT/outputs/$T.tar.gz" -C "$PROJECT_ROOT/outputs" || log "WARN: untar $T"
done
log ">>> high-rho regen (pickscore rho 320/640/1280 + images)"
mkdir -p "$PROJECT_ROOT/outputs/empty_ckpts2"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward pickscore --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/empty_ckpts2" \
    --tfg-rhos 320,640,1280 --stack-rhos "" --seed-bases 0,1000 \
    --image-dir "$PROJECT_ROOT/outputs/study_highrho_images" \
    --out "$PROJECT_ROOT/outputs/study_highrho_matrix.json" \
    || { log "FATAL: highrho regen failed"; exit 1; }
log ">>> detector on high-rho tree"
python3 "$PROJECT_ROOT/scripts/cloud/detector_score.py" \
    --image-root "$PROJECT_ROOT/outputs/study_highrho_images" \
    --prompts-file "$PROMPTS_FILE" --num-prompts 150 \
    --out "$PROJECT_ROOT/outputs/study_highrho_detector.json" || log "WARN: detector failed"
log ">>> ImageReward external scoring (all trees)"
python3 - <<'PY'
import json, pathlib, statistics, math, torch
import ImageReward as RM
from PIL import Image
rm = RM.load("ImageReward-v1.0", device="cuda")
ROOT = pathlib.Path(".")
def prompts_for(tree):
    if "ocr" in tree:
        return [l.strip() for l in open("repos/flow_grpo/dataset/ocr/test.txt") if l.strip()][:150]
    return [l.strip() for l in open("outputs/eval_prompts_pub.txt") if l.strip()][:150]
out = {}
for tree in ["study_pickscore_pub_images", "study_clip_ocr_pub_images", "study_ensemble_images", "study_highrho_images"]:
    root = ROOT / "outputs" / tree
    if not root.is_dir(): continue
    prompts = prompts_for(tree)
    out[tree] = {}
    for arm_dir in sorted(root.iterdir()):
        if not arm_dir.is_dir(): continue
        scores = {}
        for b_dir in sorted(arm_dir.iterdir()):
            row = []
            for i, prompt in enumerate(prompts):
                p = b_dir / f"{i:03d}.png"
                if not p.exists(): row.append(None); continue
                with torch.no_grad():
                    row.append(float(rm.score(prompt, str(p))))
            scores[b_dir.name] = row
        out[tree][arm_dir.name] = scores
        done = sum(1 for b in scores for v in scores[b] if v is not None)
        print(f"[ir] {tree}/{arm_dir.name}: {done} scored", flush=True)
        pathlib.Path("outputs/study_imagereward_external.json").write_text(json.dumps(out))
PY
log "=== PROBE2 done ==="
