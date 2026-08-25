#!/bin/bash
# Runs ON the box. Measures prompt-dependence of each reward model as a
# CONTINUOUS coordinate: mean pairwise cosine similarity of the reward
# gradient w.r.t. the image across prompts, at matched inputs.
# Settings mirror the map: (clip, GenEval), (clip, OCR), (pickscore, GenEval),
# (aesthetic, GenEval), (ensemble, GenEval).
# Usage: HF_TOKEN=... bash scripts/cloud/run_pdep.sh
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
python3 -m pip install --quiet --force-reinstall 'numpy==1.26.4'
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
git clone -q https://github.com/yifan123/flow_grpo repos/flow_grpo 2>/dev/null || true
python3 - <<'PY'
import json, sys, pathlib, itertools, torch
sys.path.insert(0, ".")
from scripts.cloud.run_smoke_cells import build_pipe, encode_prompt_offloaded
from scripts.clip_predictor import CLIPPromptPredictor
from scripts.pickscore_predictor import PickScorePromptPredictor
from scripts.aesthetic_predictor import AestheticPredictor
from scripts.ensemble_predictor import EnsemblePromptPredictor

device = "cuda"
gen_prompts = [d["prompt"] for d in json.load(open("scripts/cloud/prompts_smoke.json"))][:16]
ocr_prompts = [l.strip() for l in open("repos/flow_grpo/dataset/ocr/test.txt") if l.strip()][:16]

# Generate 8 base images (fast: 20 steps) from a mix of prompts
pipe = build_pipe("stabilityai/stable-diffusion-3.5-medium", device, text_encoders_cpu=False, vae_tiling=True)
pipe.set_progress_bar_config(disable=True)
imgs = []
for i in range(8):
    g = torch.Generator(device=device).manual_seed(1000 + i)
    with torch.no_grad():
        img = pipe(gen_prompts[i], num_inference_steps=20, guidance_scale=4.5,
                   height=512, width=512, generator=g).images[0]
    import torchvision.transforms.functional as TF
    imgs.append((TF.to_tensor(img).to(device) * 2 - 1).unsqueeze(0))
print("[pdep] 8 images generated", flush=True)
del pipe; torch.cuda.empty_cache()

def grad_sim(predictor, prompts):
    """Mean pairwise cosine similarity of d reward / d image across prompts."""
    sims_per_img = []
    for x0 in imgs:
        grads = []
        for p in prompts:
            x = x0.clone().float().requires_grad_(True)
            s = predictor(x, [p])
            g = torch.autograd.grad(s.sum(), x)[0].flatten()
            grads.append(g / (g.norm() + 1e-12))
        G = torch.stack(grads)
        cos = G @ G.T
        n = len(prompts)
        off = (cos.sum() - torch.diagonal(cos).sum()) / (n * n - n)
        sims_per_img.append(float(off))
    import statistics
    return statistics.mean(sims_per_img), statistics.stdev(sims_per_img)

out = {}
settings = [
    ("clip_geneval", CLIPPromptPredictor, gen_prompts),
    ("clip_ocr", CLIPPromptPredictor, ocr_prompts),
    ("pickscore_geneval", PickScorePromptPredictor, gen_prompts),
    ("aesthetic", AestheticPredictor, gen_prompts),
    ("ensemble_geneval", EnsemblePromptPredictor, gen_prompts),
]
for name, cls, prompts in settings:
    pred = cls(device=device)
    m, sd = grad_sim(pred, prompts)
    out[name] = {"mean_pairwise_cos": m, "sd_over_images": sd, "n_prompts": len(prompts), "n_images": len(imgs)}
    print(f"[pdep] {name}: cos={m:.4f} sd={sd:.4f}", flush=True)
    del pred; torch.cuda.empty_cache()
pathlib.Path("outputs/study_pdep.json").write_text(json.dumps(out, indent=1))
PY
log "=== PDEP done ==="
