#!/bin/bash
# Runs ON the box. BLIP-ITM external scoring (BLIP family, cross-attention ITM
# head — non-contrastive judge) over the saved trees. Replaces the ImageReward
# package, whose bundled BLIP is incompatible with modern transformers.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
git clone -q https://github.com/yifan123/flow_grpo repos/flow_grpo 2>/dev/null || true
log() { echo "[$(date +%H:%M:%S)] $*"; }
python3 -m pip install --quiet --force-reinstall 'transformers==4.49.0' 'tokenizers>=0.20' 'huggingface_hub>=0.23' >/dev/null 2>&1 || true
python3 -m pip install --quiet --force-reinstall 'numpy==1.26.4' >/dev/null 2>&1 || true
python3 - <<'PY'
import json, pathlib, torch
from transformers import BlipProcessor, BlipForImageTextRetrieval
from PIL import Image
device = "cuda"
proc = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco", torch_dtype=torch.float16).to(device).eval()
def score(img, prompt):
    inp = proc(images=img, text=prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        out = model(pixel_values=inp["pixel_values"].half(), input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], use_itm_head=True)
    return float(torch.softmax(out.itm_score.float(), dim=-1)[0, 1].item())
def prompts_for(tree):
    if "ocr" in tree:
        return [l.strip() for l in open("repos/flow_grpo/dataset/ocr/test.txt") if l.strip()][:150]
    return [l.strip() for l in open("outputs/eval_prompts_pub.txt") if l.strip()][:150]
out = {}
for tree in ["study_pickscore_pub_images", "study_clip_ocr_pub_images", "study_ensemble_images", "study_highrho_images"]:
    root = pathlib.Path("outputs") / tree
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
                try: row.append(score(Image.open(p).convert("RGB"), prompt))
                except Exception: row.append(None)
            scores[b_dir.name] = row
        out[tree][arm_dir.name] = scores
        print(f"[ir] {tree}/{arm_dir.name} scored", flush=True)
        pathlib.Path("outputs/study_imagereward_external.json").write_text(json.dumps(out))
PY
log "=== IRONLY done ==="
