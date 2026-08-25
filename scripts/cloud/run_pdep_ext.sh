#!/bin/bash
# Runs ON the box. Extends the prompt-dependence coordinate to two more reward
# models (HPSv2, ImageReward) to populate the diagnostic axis beyond its two
# anchors. Eval-only, cheap.
#
# HARD-WON ORDERING: both open_clip_torch AND image-reward perturb transformers'
# shared deps (huggingface_hub / tokenizers), which breaks the diffusers SD3
# pipeline import (CLIPTextModelWithProjection). So we generate + save the 8 base
# images using ONLY the proven diffusers+transformers recipe, THEN install each
# fragile predictor lib and score it on the SAVED images (no diffusers import
# afterward). Each predictor is smoke-gated: a load failure is recorded, not fatal.
# Usage: HF_TOKEN=... bash scripts/cloud/run_pdep_ext.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
IMGS="$PROJECT_ROOT/outputs/pdep_ext_imgs.pt"
JSON="$PROJECT_ROOT/outputs/study_pdep_ext.json"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?}"

# ---------- deps: ONLY the proven diffusers+transformers recipe (no open_clip/IR) ----------
log ">>> deps (proven diffusers + transformers only)"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet 'diffusers>=0.31,<0.40' 'transformers==4.49.0' \
    'peft>=0.12' 'accelerate>=0.33' sentencepiece protobuf safetensors torchvision tqdm
python3 -m pip install --quiet --force-reinstall 'numpy==1.26.4'
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"

# ---------- Step A: generate + save 8 images (clean env), seed the JSON ----------
log ">>> step A: generate + save base images"
python3 - <<'PY'
import json, sys, pathlib, torch
sys.path.insert(0, ".")
from scripts.cloud.run_smoke_cells import build_pipe
import torchvision.transforms.functional as TF
device = "cuda"
gen_prompts = [d["prompt"] for d in json.load(open("scripts/cloud/prompts_smoke.json"))][:16]
pipe = build_pipe("stabilityai/stable-diffusion-3.5-medium", device, text_encoders_cpu=False, vae_tiling=True)
pipe.set_progress_bar_config(disable=True)
imgs = []
for i in range(8):
    g = torch.Generator(device=device).manual_seed(1000 + i)
    with torch.no_grad():
        img = pipe(gen_prompts[i], num_inference_steps=20, guidance_scale=4.5,
                   height=512, width=512, generator=g).images[0]
    imgs.append((TF.to_tensor(img).to(device) * 2 - 1).unsqueeze(0).detach().cpu())
torch.save({"imgs": imgs, "prompts": gen_prompts}, "outputs/pdep_ext_imgs.pt")
pathlib.Path("outputs/study_pdep_ext.json").write_text(json.dumps({}, indent=1))
print(f"[pdep-ext] saved {len(imgs)} images", flush=True)
PY
[ -f "$IMGS" ] || { log "FATAL: image generation failed, no $IMGS"; exit 1; }

# Shared measurement helper written once; each predictor step imports it.
cat > "$PROJECT_ROOT/outputs/_pdep_measure.py" <<'PY'
import json, sys, statistics, pathlib, torch
sys.path.insert(0, ".")
device = "cuda"
def run(name, loader):
    data = torch.load("outputs/pdep_ext_imgs.pt")
    imgs, prompts = data["imgs"], data["prompts"]
    out = json.loads(pathlib.Path("outputs/study_pdep_ext.json").read_text())
    try:
        pred = loader()
        s0 = pred(imgs[0].float().to(device), [prompts[0]])
        assert torch.isfinite(s0).all()
        print(f"[pdep-ext] {name} loaded OK (smoke {float(s0):.4f})", flush=True)
        sims = []
        for x0 in imgs:
            grads = []
            for p in prompts:
                x = x0.clone().float().to(device).requires_grad_(True)
                g = torch.autograd.grad(pred(x, [p]).sum(), x)[0].flatten()
                grads.append(g / (g.norm() + 1e-12))
            G = torch.stack(grads); cos = G @ G.T; n = len(prompts)
            sims.append(float((cos.sum() - torch.diagonal(cos).sum()) / (n * n - n)))
        m, sd = statistics.mean(sims), statistics.stdev(sims)
        out[name] = {"mean_pairwise_cos": m, "sd_over_images": sd,
                     "n_prompts": len(prompts), "n_images": len(imgs), "status": "ok"}
        print(f"[pdep-ext] {name}: cos={m:.4f} sd={sd:.4f}", flush=True)
    except Exception as e:
        out[name] = {"status": "load_failed", "error": f"{type(e).__name__}: {e}"}
        print(f"[pdep-ext] {name} FAILED: {type(e).__name__}: {e}", flush=True)
    pathlib.Path("outputs/study_pdep_ext.json").write_text(json.dumps(out, indent=1))
PY

# ---------- Step B: HPSv2 (open_clip) ----------
log ">>> step B: install open_clip + HPSv2 coordinate"
python3 -m pip install --quiet open_clip_torch ftfy regex || log "WARN: open_clip install nonzero"
python3 - <<'PY'
import sys; sys.path.insert(0, "outputs")
from _pdep_measure import run
def load():
    from scripts.hpsv2_predictor import HPSv2PromptPredictor
    return HPSv2PromptPredictor(device="cuda")
run("hpsv2_geneval", load)
PY

# ---------- Step C: ImageReward (BLIP) ----------
log ">>> step C: install image-reward + ImageReward coordinate"
python3 -m pip install --quiet image-reward || log "WARN: image-reward install nonzero"
python3 - <<'PY'
import sys; sys.path.insert(0, "outputs")
from _pdep_measure import run
def load():
    from scripts.imagereward_predictor import ImageRewardPredictor
    return ImageRewardPredictor(device="cuda")
run("imagereward_geneval", load)
PY

log "=== PDEP-EXT done ==="
