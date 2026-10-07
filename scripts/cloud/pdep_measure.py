#!/usr/bin/env python3
"""
Pre-flight coordinate 2: prompt-dependence of a reward, measured as the mean
pairwise cosine similarity of d reward / d image across prompts at matched
images (8 base images x 16 prompts). 1.0 = prompt-independent (aesthetic),
lower = the gradient direction depends on the prompt.

Standalone version of the inline snippet in run_pdep.sh so new rewards can be
added by name. Images are generated once (bf16 pipe) and scored in fp32.

Usage (on the box):
    python3 scripts/cloud/pdep_measure.py --settings hpsv2:geneval,siglip:geneval,clip:pickapic \
        --out outputs/study_pdep_rw8.json
Setting = <reward>:<prompt_pool>, prompt_pool in {geneval, ocr, pickapic}.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

POOLS = {
    "geneval": ("scripts/cloud/prompts_smoke.json", "json"),
    "ocr": ("repos/flow_grpo/dataset/ocr/test.txt", "txt"),
    "pickapic": ("repos/flow_grpo/dataset/pickscore/test.txt", "txt"),
}


def load_pool(name: str, n: int) -> list[str]:
    path, kind = POOLS[name]
    p = PROJECT_ROOT / path
    if kind == "json":
        lines = [d["prompt"] for d in json.loads(p.read_text())]
    else:
        lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    return lines[:n]


def make_predictor(reward: str, device: str):
    if reward == "clip":
        from scripts.clip_predictor import CLIPPromptPredictor
        return CLIPPromptPredictor(device=device)
    if reward == "pickscore":
        from scripts.pickscore_predictor import PickScorePromptPredictor
        return PickScorePromptPredictor(device=device)
    if reward == "aesthetic":
        from scripts.aesthetic_predictor import AestheticPredictor
        return AestheticPredictor(device=device)
    if reward == "ensemble":
        from scripts.ensemble_predictor import EnsemblePromptPredictor
        return EnsemblePromptPredictor(device=device)
    if reward == "hpsv2":
        from scripts.hpsv2_predictor import HPSv2PromptPredictor
        return HPSv2PromptPredictor(device=device)
    if reward == "siglip":
        from scripts.siglip_predictor import SiglipPromptPredictor
        return SiglipPromptPredictor(device=device)
    raise ValueError(reward)


def grad_sim(predictor, prompts: list[str], imgs: list[torch.Tensor]) -> tuple[float, float]:
    sims = []
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
        sims.append(float((cos.sum() - torch.diagonal(cos).sum()) / (n * n - n)))
    return statistics.mean(sims), (statistics.stdev(sims) if len(sims) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, help="comma list of <reward>:<pool>")
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="stabilityai/stable-diffusion-3.5-medium")
    args = ap.parse_args()
    device = "cuda"

    from scripts.cloud.run_smoke_cells import build_pipe
    import torchvision.transforms.functional as TF

    gen_prompts = load_pool("geneval", args.n_images)
    # Text encoders stay on GPU: with them on CPU the SD3 pipeline's encode_prompt
    # sends input_ids to cuda and T5's embedding crashes (cpu vs cuda). SD3.5-M
    # bf16 is ~16GB fully resident, fine on 40GB; the pipe is freed before the
    # predictors load.
    pipe = build_pipe(args.base, device, text_encoders_cpu=False, vae_tiling=True)
    pipe.set_progress_bar_config(disable=True)
    imgs = []
    for i in range(args.n_images):
        g = torch.Generator(device=device).manual_seed(1000 + i)
        with torch.no_grad():
            img = pipe(gen_prompts[i], num_inference_steps=20, guidance_scale=4.5,
                       height=512, width=512, generator=g).images[0]
        imgs.append((TF.to_tensor(img).to(device).float() * 2 - 1).unsqueeze(0))
    print(f"[pdep] {len(imgs)} images generated", flush=True)
    del pipe
    torch.cuda.empty_cache()

    out_path = Path(args.out)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for spec in [s for s in args.settings.split(",") if s]:
        reward, pool = spec.split(":")
        key = f"{reward}_{pool}"
        prompts = load_pool(pool, args.n_prompts)
        pred = make_predictor(reward, device)
        m, sd = grad_sim(pred, prompts, imgs)
        out[key] = {"mean_pairwise_cos": m, "sd_over_images": sd,
                    "n_prompts": len(prompts), "n_images": len(imgs)}
        print(f"[pdep] {key}: cos={m:.4f} sd={sd:.4f}", flush=True)
        out_path.write_text(json.dumps(out, indent=1))
        del pred
        torch.cuda.empty_cache()
    print("[pdep] done", flush=True)


if __name__ == "__main__":
    main()
