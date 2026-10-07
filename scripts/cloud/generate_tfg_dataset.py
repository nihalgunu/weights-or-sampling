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


def load_prompt_pool(path) -> list[dict]:
    """json list of {id,prompt}, jsonl with a "prompt" key (flow_grpo geneval
    train_metadata.jsonl, 50k GenEval-style prompts), or one prompt per line."""
    p = Path(path)
    if p.suffix == ".json":
        data = json.load(open(p))
        items = data if isinstance(data, list) else data.get("prompts", [])
        out = []
        for i, d in enumerate(items):
            if isinstance(d, dict):
                out.append({"id": d.get("id", i), "prompt": d["prompt"], "tag": d.get("tag", "")})
            else:
                out.append({"id": i, "prompt": str(d), "tag": ""})
        return out
    if p.suffix == ".jsonl":
        out = []
        for i, ln in enumerate(open(p)):
            ln = ln.strip()
            if ln:
                d = json.loads(ln)
                out.append({"id": i, "prompt": d["prompt"], "tag": d.get("tag", "")})
        return out
    return [{"id": i, "prompt": ln.strip(), "tag": ""} for i, ln in enumerate(open(p)) if ln.strip()]


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
    parser.add_argument("--tfg_mu", type=float, default=0.0,
                        help="x0-refinement branch strength (0 = rho-only teacher, the old probe).")
    parser.add_argument("--apply_range", default="0.2,0.8",
                        help="sigma range in which guidance fires; default matches the eval TFG arms.")
    parser.add_argument("--exclude_file", default=None,
                        help="Prompt file (txt/json/jsonl) whose prompts are removed from the pool (the eval set).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Total teacher samples to produce (prompts x seeds), e.g. 5000.")
    parser.add_argument("--seeds_per_prompt", type=int, default=1,
                        help="Teacher rollouts per prompt (different noise), for small prompt pools.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    (output_dir / "latents").mkdir(parents=True, exist_ok=True)

    prompts = load_prompt_pool(args.prompts_file)
    if args.exclude_file:
        excl = set(p["prompt"] for p in load_prompt_pool(args.exclude_file))
        before = len(prompts)
        prompts = [p for p in prompts if p["prompt"] not in excl]
        print(f"excluded {before - len(prompts)} prompts present in {args.exclude_file}")
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    jobs = []
    for p in prompts:
        for k in range(args.seeds_per_prompt):
            jobs.append({"id": p["id"] * args.seeds_per_prompt + k, "prompt": p["prompt"],
                         "tag": p.get("tag", ""), "seed_k": k})
    if args.max_samples:
        jobs = jobs[: args.max_samples]
    print(f"{len(prompts)} prompts -> {len(jobs)} teacher jobs")

    print(f"Loading SD3.5-M for teacher rollouts (text_encoders_cpu={args.text_encoders_cpu})...")
    pipe = build_pipe(
        args.model, device,
        text_encoders_cpu=args.text_encoders_cpu,
        vae_tiling=args.vae_tiling,
    )
    predictor = CLIPPromptPredictor(device=device, torch_dtype=torch.float32)
    exec_device = next(pipe.transformer.parameters()).device

    lo, hi = (float(v) for v in args.apply_range.split(","))
    guidance = TFGFlowGuidance(
        predictor=predictor,
        guidance_scale=args.tfg_rho,
        recurrence_steps=1,
        apply_every_n_steps=args.tfg_every_n,
        apply_range=(lo, hi),
        mu=args.tfg_mu,
    )
    patch_scheduler_step(pipe, guidance)

    manifest = {
        "model": args.model,
        "tfg_rho": args.tfg_rho,
        "tfg_mu": args.tfg_mu,
        "apply_range": [lo, hi],
        "tfg_every_n": args.tfg_every_n,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed_base": args.seed_base,
        "items": [],
    }
    manifest_path = output_dir / "manifest.json"
    done_ids = set()
    if manifest_path.exists():
        try:
            prev = json.load(open(manifest_path))
            for it in prev.get("items", []):
                if (output_dir / it["latent_path"]).exists():
                    manifest["items"].append(it)
                    done_ids.add(it["id"])
            if done_ids:
                print(f"resuming: {len(done_ids)} teacher samples already present")
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not read previous manifest ({e}); starting fresh")

    try:
        t0 = time.time()
        n_new = 0
        for p in tqdm(jobs, desc="teacher rollouts"):
            if p["id"] in done_ids:
                continue
            seed = args.seed_base + p["id"]
            latent = generate_teacher_latent(
                pipe, guidance, p["prompt"], seed,
                args.num_inference_steps, args.guidance_scale,
                args.height, args.width, exec_device,
            )
            latent_path = output_dir / "latents" / f"{p['id']:06d}.pt"
            torch.save(latent, latent_path)
            manifest["items"].append({
                "id": p["id"],
                "prompt": p["prompt"],
                "tag": p.get("tag", ""),
                "seed": seed,
                "latent_path": str(latent_path.relative_to(output_dir)),
                "fires": guidance.fires_this_run,
            })
            n_new += 1
            # Persist the manifest periodically so a crash loses little; Stage 2
            # resumes from whatever landed.
            if n_new % 25 == 0:
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
