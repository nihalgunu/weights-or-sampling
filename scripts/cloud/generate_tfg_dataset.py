#!/usr/bin/env python3
"""
Stage 1 of TFG Distillation: generate teacher latents via TFG-guided rollout.

For each prompt in a train pool, run the base SD3.5-M pipeline with our
TFG-Flow hook active at a chosen ρ, and save the *final latent* (not the
decoded image). Saving latents instead of images skips the VAE encode roundtrip
at training time — lossless and faster. One .pt file per prompt + a manifest
JSON tying them to prompt text.

This is the most expensive stage (each sample is a full TFG denoising run,
~15-18s on A100). Budget for N=200 samples at 15 s/each = 50 min.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.clip_predictor import CLIPPromptPredictor
from scripts.tfg_flow import (
    PromptContext,
    TFGFlowGuidance,
    patch_scheduler_step,
    unpatch_scheduler_step,
)
from scripts.cloud.run_smoke_cells import (
    build_pipe,
    encode_prompt_offloaded,
)


def generate_teacher_latent(
    pipe,
    guidance: TFGFlowGuidance,
    prompt_text: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    exec_device: torch.device,
) -> torch.Tensor:
    """One TFG-guided rollout. Returns the post-denoising latent tensor, CPU."""
    gen = torch.Generator(device=exec_device).manual_seed(seed)
    pos, neg, pos_p, neg_p = encode_prompt_offloaded(pipe, prompt_text, str(exec_device))
    guidance.set_prompt_context(
        PromptContext(
            prompt_embeds=pos,
            pooled_prompt_embeds=pos_p,
            prompt_text=[prompt_text],
        )
    )
    # output_type="latent" returns the final latent instead of decoding — avoids
    # the encode/decode roundtrip and saves wall time per sample.
    out = pipe(
        prompt_embeds=pos,
        negative_prompt_embeds=neg,
        pooled_prompt_embeds=pos_p,
        negative_pooled_prompt_embeds=neg_p,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=gen,
        output_type="latent",
    )
    # SD3 returns latents in the pipeline's "scaled" form — keep as-is; training
    # will sample noise and mix using the same scheduler convention.
    latent = out.images[0] if hasattr(out, "images") else out[0]
    return latent.detach().to("cpu", dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", default="scripts/cloud/prompts_train.json")
    parser.add_argument("--output_dir", default="outputs/distill_dataset")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--tfg_rho", type=float, default=10.0,
                        help="Teacher's guidance strength. ρ=10 gave Δ=+0.022 on "
                             "the base sweep — strong enough to imprint, weak "
                             "enough to avoid obvious failure modes at ρ=20.")
    parser.add_argument("--tfg_every_n", type=int, default=1,
                        help="Fire TFG every N steps. 1 = every step.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed_base", type=int, default=1000,
                        help="Base seed; per-prompt seed is seed_base + prompt_id. "
                             "Chosen not to overlap with the eval seed 42 range.")
    parser.add_argument("--text_encoders_cpu", action="store_true", default=True)
    parser.add_argument("--vae_tiling", action="store_true", default=True)
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Cap for quick dry-runs; default processes all.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    (output_dir / "latents").mkdir(parents=True, exist_ok=True)

    with open(args.prompts_file) as f:
        prompts = json.load(f)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]

    print(f"Loading SD3.5-M for teacher rollouts (text_encoders_cpu={args.text_encoders_cpu})...")
    pipe = build_pipe(
        args.model, device,
        text_encoders_cpu=args.text_encoders_cpu,
        vae_tiling=args.vae_tiling,
    )
    predictor = CLIPPromptPredictor(device=device, torch_dtype=torch.float32)
    exec_device = next(pipe.transformer.parameters()).device

    guidance = TFGFlowGuidance(
        predictor=predictor,
        guidance_scale=args.tfg_rho,
        recurrence_steps=1,
        apply_every_n_steps=args.tfg_every_n,
        apply_range=(0.1, 0.9),
    )
    patch_scheduler_step(pipe, guidance)

    manifest = {
        "model": args.model,
        "tfg_rho": args.tfg_rho,
        "tfg_every_n": args.tfg_every_n,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed_base": args.seed_base,
        "items": [],
    }
    manifest_path = output_dir / "manifest.json"

    try:
        t0 = time.time()
        for p in tqdm(prompts, desc="teacher rollouts"):
            seed = args.seed_base + p["id"]
            latent = generate_teacher_latent(
                pipe, guidance, p["prompt"], seed,
                args.num_inference_steps, args.guidance_scale,
                args.height, args.width, exec_device,
            )
            latent_path = output_dir / "latents" / f"{p['id']:04d}.pt"
            torch.save(latent, latent_path)
            manifest["items"].append({
                "id": p["id"],
                "prompt": p["prompt"],
                "tag": p.get("tag", ""),
                "seed": seed,
                "latent_path": str(latent_path.relative_to(output_dir)),
                "fires": guidance.fires_this_run,
            })
            # Persist manifest every sample so a crash mid-generation doesn't
            # lose progress — Stage 2 can resume from whatever landed.
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        manifest["wall_s"] = time.time() - t0
    finally:
        unpatch_scheduler_step(pipe)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest['items'])} teacher latents in {manifest.get('wall_s', 0):.0f}s")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
