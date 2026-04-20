#!/usr/bin/env python3
"""
ρ-sweep: Cell B at ρ ∈ {2, 5, 10, 20}, apply_every_n=1.

Purpose: the first smoke run showed TFG plumbing works but at ρ=1.0 the effect
on CLIP score was within noise (|Δ| < 0.002). This script asks whether cranking
ρ past 1 produces a monotonic response — positive slope means TFG is steering
toward the predictor; null slope means the guidance signal is flat; negative
slope means it's degrading output. Any of these is a real finding.

apply_every_n=1 (fire on every step) makes the ρ effect maximally visible.

Loads SD3.5-M + CLIP predictor once, generates Cell A (baseline) + one Cell B
variant per ρ. Saves per-ρ json + paired-delta summary.
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
from scripts.cloud.run_smoke_cells import (  # reuse helpers
    build_pipe,
    encode_prompt_offloaded,
    load_prompts,
    score_image,
)


def run_cell_variant(
    pipe,
    predictor,
    prompts: list[dict],
    output_dir: Path,
    cell_tag: str,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
    tfg_scale: float | None,
    tfg_every_n: int,
) -> dict:
    """Generate one cell. tfg_scale=None means baseline (no TFG)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "cell": cell_tag,
        "tfg_scale": tfg_scale,
        "tfg_every_n": tfg_every_n if tfg_scale is not None else None,
        "items": [],
    }

    guidance = None
    if tfg_scale is not None:
        guidance = TFGFlowGuidance(
            predictor=predictor,
            guidance_scale=tfg_scale,
            recurrence_steps=1,
            apply_every_n_steps=tfg_every_n,
            apply_range=(0.1, 0.9),
        )
        patch_scheduler_step(pipe, guidance)

    exec_device = next(pipe.transformer.parameters()).device

    try:
        t0 = time.time()
        for p in tqdm(prompts, desc=f"cell {cell_tag}"):
            prompt_text = p["prompt"]
            gen = torch.Generator(device=exec_device).manual_seed(seed + p["id"])
            pos_embeds, neg_embeds, pos_pooled, neg_pooled = encode_prompt_offloaded(
                pipe, prompt_text, str(exec_device)
            )
            if guidance is not None:
                guidance.set_prompt_context(
                    PromptContext(
                        prompt_embeds=pos_embeds,
                        pooled_prompt_embeds=pos_pooled,
                        prompt_text=[prompt_text],
                    )
                )
            out = pipe(
                prompt_embeds=pos_embeds,
                negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=pos_pooled,
                negative_pooled_prompt_embeds=neg_pooled,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=gen,
            )
            image = out.images[0]
            image_path = output_dir / f"{p['id']:03d}.png"
            image.save(image_path)
            score = score_image(predictor, image, prompt_text)
            item = {
                "id": p["id"],
                "prompt": prompt_text,
                "category": p.get("category"),
                "image_path": str(image_path),
                "clip_score": score,
            }
            if guidance is not None:
                item["tfg_fires"] = guidance.fires_this_run
            results["items"].append(item)
        results["wall_time_s"] = time.time() - t0
        results["mean_clip_score"] = sum(i["clip_score"] for i in results["items"]) / len(results["items"])
    finally:
        if guidance is not None:
            unpatch_scheduler_step(pipe)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", default="scripts/cloud/prompts_smoke.json")
    parser.add_argument("--output_dir", default="outputs/sweep")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rhos", default="2,5,10,20",
                        help="Comma-separated ρ values for Cell B variants.")
    parser.add_argument("--apply_every_n", type=int, default=1,
                        help="Fire TFG on every N-th step. 1 = every step (max signal).")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip Cell A if it already exists.")
    parser.add_argument("--text_encoders_cpu", action="store_true", default=True)
    parser.add_argument("--vae_tiling", action="store_true", default=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a LoRA adapter to load on top of the base model. "
                             "When set, cell_a semantically becomes Cell C and cell_b_rho* "
                             "becomes Cell D variants.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.prompts_file))
    rho_list = [float(x) for x in args.rhos.split(",")]

    print(f"Loading SD3.5-M (text_encoders_cpu={args.text_encoders_cpu})...")
    pipe = build_pipe(
        args.model, device,
        text_encoders_cpu=args.text_encoders_cpu,
        vae_tiling=args.vae_tiling,
    )
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if (ckpt / "adapter_model.safetensors").exists() or (ckpt / "adapter_model.bin").exists():
            print(f"Loading LoRA checkpoint from {ckpt}")
            pipe.load_lora_weights(str(ckpt))
        else:
            print(f"WARN: checkpoint path {ckpt} has no adapter_model.* — skipping load")
    predictor = CLIPPromptPredictor(device=device, torch_dtype=torch.float32)

    all_results = {}

    # Baseline (Cell A)
    baseline_path = output_dir / "cell_a_results.json"
    if not args.skip_baseline or not baseline_path.exists():
        res_a = run_cell_variant(
            pipe, predictor, prompts,
            output_dir / "cell_a", "A",
            args.num_inference_steps, args.guidance_scale,
            args.height, args.width, args.seed,
            tfg_scale=None, tfg_every_n=1,
        )
        with open(baseline_path, "w") as f:
            json.dump(res_a, f, indent=2)
        print(f"Cell A mean CLIP = {res_a['mean_clip_score']:.4f}  ({res_a['wall_time_s']:.0f}s)")
        all_results["A"] = res_a
    else:
        with open(baseline_path) as f:
            all_results["A"] = json.load(f)
        print(f"Cell A reused: mean CLIP = {all_results['A']['mean_clip_score']:.4f}")

    # Sweep
    for rho in rho_list:
        tag = f"B_rho{rho:g}"
        res = run_cell_variant(
            pipe, predictor, prompts,
            output_dir / f"cell_b_rho{rho:g}", tag,
            args.num_inference_steps, args.guidance_scale,
            args.height, args.width, args.seed,
            tfg_scale=rho, tfg_every_n=args.apply_every_n,
        )
        with open(output_dir / f"cell_b_rho{rho:g}_results.json", "w") as f:
            json.dump(res, f, indent=2)
        all_results[tag] = res
        delta = res["mean_clip_score"] - all_results["A"]["mean_clip_score"]
        fires_avg = sum(i.get("tfg_fires", 0) for i in res["items"]) / len(res["items"])
        print(f"  ρ={rho:5.1f}  mean CLIP = {res['mean_clip_score']:.4f}  "
              f"Δ={delta:+.4f}  fires/prompt={fires_avg:.1f}  ({res['wall_time_s']:.0f}s)")

    # Summary
    summary = {
        "baseline_mean_clip": all_results["A"]["mean_clip_score"],
        "rho_list": rho_list,
        "apply_every_n": args.apply_every_n,
        "per_rho": [
            {
                "rho": rho,
                "mean_clip": all_results[f"B_rho{rho:g}"]["mean_clip_score"],
                "delta_vs_baseline": all_results[f"B_rho{rho:g}"]["mean_clip_score"] - all_results["A"]["mean_clip_score"],
                "wall_s": all_results[f"B_rho{rho:g}"]["wall_time_s"],
            }
            for rho in rho_list
        ],
    }
    with open(output_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== ρ-SWEEP SUMMARY ===")
    print(f"Cell A baseline: {summary['baseline_mean_clip']:.4f}")
    for r in summary["per_rho"]:
        sign = "+" if r["delta_vs_baseline"] >= 0 else ""
        print(f"  ρ={r['rho']:5.1f}  CLIP={r['mean_clip']:.4f}  Δ={sign}{r['delta_vs_baseline']:.4f}")


if __name__ == "__main__":
    main()
