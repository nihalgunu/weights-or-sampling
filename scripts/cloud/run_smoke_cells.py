#!/usr/bin/env python3
"""
Smoke run for Cells A + B on a Lambda A100.

Cell A: base SD3.5-M, 30 prompts, 512², 28 steps, no TFG.
Cell B: same + TFG (rho=1.0, every 4 steps, CLIP predictor).

Primary goal: catch GPU-only bugs in scripts/tfg_flow.py and scripts/clip_predictor.py
against a real SD3 pipeline. Real SD3 has 16-channel latents, real VAE constants,
real transformer shapes — the CPU smoke test stubs don't cover those.

Secondary goal: first direction signal on whether TFG+CLIP nudges mean CLIP score.
A negative or null result is informative — it means the hook fires but guidance
is weak at these settings. A positive result is not proof of GenEval gain (different
metric) but proof the pipeline produces the right *kind* of result.

Scoring: each image is scored with CLIPPromptPredictor against its prompt. We
report mean + std across the 30 prompts per cell. No GenEval in this smoke —
mmdetection install is a separate risk and we defer it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
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


def load_prompts(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_pipe(
    model_name: str,
    device: str,
    text_encoders_cpu: bool = False,
    vae_tiling: bool = False,
):
    """Build the SD3 pipeline with optional memory mitigations.

    text_encoders_cpu: after loading, move all text encoders (CLIP-L, CLIP-G,
    T5-XXL) permanently to CPU. Saves ~11.5GB on SD3.5-M. Transformer + VAE
    stay on GPU, which is what TFG needs — our hook retains VAE activations
    through backward, which is incompatible with `enable_model_cpu_offload`
    (that hook moves modules back to CPU right after forward, leaving weights
    and activations on different devices for autograd.grad). Text encoders
    run on CPU during encode_prompt (slow but fine for 30 prompts).

    vae_tiling decodes the latent in chunks so peak VAE decode memory scales
    with tile size rather than full image. Safety belt at 512².
    """
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)

    if text_encoders_cpu:
        # Only offload T5-XXL (text_encoder_3) — it's ~9.5GB bf16 and dominates
        # memory. CLIP-L (text_encoder, ~0.6GB) and CLIP-G (text_encoder_2, ~1.4GB)
        # stay on GPU so `pipe.device` still reports CUDA and `prepare_latents`
        # allocates on the right device. Offloading ALL three makes pipe.device
        # return CPU, which breaks pipe's internal prepare_latents/exec_device.
        t5 = getattr(pipe, "text_encoder_3", None)
        if t5 is not None:
            t5.to("cpu")
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"  [mem] after T5 offload: {free/1e9:.1f}GB free of {total/1e9:.1f}GB")

    if vae_tiling:
        pipe.vae.enable_tiling()
    return pipe


def encode_prompt_offloaded(pipe, prompt: str, device: str):
    """Encode a prompt, moving T5 to GPU just for its pass and back to CPU after.

    CLIP-L and CLIP-G stay on GPU permanently (they're small). T5-XXL (~9.5GB)
    is the only text encoder we shuttle. Per-prompt shuttle cost is ~1-2s for T5.
    All returned embeds land on `device` (cuda).

    If text_encoder_3 isn't present (not all SD3 checkpoints ship T5), this
    degenerates to a normal encode_prompt call.
    """
    t5 = getattr(pipe, "text_encoder_3", None)
    t5_was_on_cpu = t5 is not None and next(t5.parameters()).device.type == "cpu"
    if t5_was_on_cpu:
        t5.to(device)
    try:
        with torch.no_grad():
            pos, neg, pos_p, neg_p = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                prompt_3=None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
    finally:
        if t5_was_on_cpu and t5 is not None:
            t5.to("cpu")
            torch.cuda.empty_cache()
    return pos, neg, pos_p, neg_p


def score_image(
    predictor: CLIPPromptPredictor, pil_image: Image.Image, prompt: str
) -> float:
    """Score a PIL image against its prompt. Runs under no_grad — smoke reporting only.

    PIL -> [-1, 1] tensor to match CLIPPromptPredictor's expected input domain
    (same as what VAE.decode produces during generation / guidance).
    """
    import torchvision.transforms.functional as TF

    img = TF.to_tensor(pil_image).to(predictor.device, dtype=predictor.dtype)
    img = img * 2.0 - 1.0  # [0,1] -> [-1,1]
    img = img.unsqueeze(0)
    with torch.no_grad():
        s = predictor(img, [prompt])
    return float(s.item())


def run_cell(
    pipe,
    predictor: CLIPPromptPredictor,
    prompts: list[dict],
    output_dir: Path,
    cell_name: str,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
    use_tfg: bool,
    tfg_scale: float = 1.0,
    tfg_every_n: int = 4,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"cell": cell_name, "use_tfg": use_tfg, "items": []}

    guidance = None
    if use_tfg:
        guidance = TFGFlowGuidance(
            predictor=predictor,
            guidance_scale=tfg_scale,
            recurrence_steps=1,
            apply_every_n_steps=tfg_every_n,
            apply_range=(0.1, 0.9),
        )
        patch_scheduler_step(pipe, guidance)

    # Execution device: transformer's current device is authoritative even after
    # we've moved text encoders to CPU. pipe.device is a heuristic that picks
    # whichever component it checks first and can report CPU when text encoders
    # were moved, which then makes pipe allocate init latents on CPU → conv2d
    # device mismatch. Using transformer.device sidesteps that trap.
    exec_device = next(pipe.transformer.parameters()).device
    assert exec_device.type == "cuda", f"transformer must be on cuda, got {exec_device}"

    try:
        t0 = time.time()
        for p in tqdm(prompts, desc=f"cell {cell_name}"):
            prompt_text = p["prompt"]

            gen = torch.Generator(device=exec_device).manual_seed(seed + p["id"])

            pos_embeds, neg_embeds, pos_pooled, neg_pooled = encode_prompt_offloaded(
                pipe, prompt_text, str(exec_device)
            )

            if use_tfg and guidance is not None:
                guidance.set_prompt_context(
                    PromptContext(
                        prompt_embeds=pos_embeds,
                        pooled_prompt_embeds=pos_pooled,
                        prompt_text=[prompt_text],
                    )
                )

            # Pass pre-computed embeds so pipe() skips its internal encode_prompt
            # (which would try to run text encoders that are currently on CPU).
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
            if use_tfg and guidance is not None:
                item["tfg_fires"] = guidance.fires_this_run
            results["items"].append(item)

        total = time.time() - t0
        results["wall_time_s"] = total
        results["mean_clip_score"] = sum(i["clip_score"] for i in results["items"]) / len(results["items"])
    finally:
        if use_tfg:
            unpatch_scheduler_step(pipe)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", default="scripts/cloud/prompts_smoke.json")
    parser.add_argument("--output_dir", default="outputs/smoke")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tfg_scale", type=float, default=1.0)
    parser.add_argument("--tfg_every_n", type=int, default=4)
    parser.add_argument("--text_encoders_cpu", action="store_true",
                        help="Move text encoders to CPU after load. Saves ~11GB. "
                             "Use on 24GB cards. Incompatible with enable_model_cpu_offload "
                             "(that one breaks autograd.grad through VAE/transformer).")
    parser.add_argument("--vae_tiling", action="store_true",
                        help="Enable pipe.vae.enable_tiling. Extra OOM safety belt.")
    parser.add_argument("--cells", default="AB", choices=["A", "B", "AB"],
                        help="Which cells to run. 'B' skips A (use when A is already saved).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARN: no CUDA — smoke will be painfully slow.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.prompts_file))
    print(f"Loaded {len(prompts)} prompts")

    print(f"Loading SD3.5-M (text_encoders_cpu={args.text_encoders_cpu}, vae_tiling={args.vae_tiling})...")
    pipe = build_pipe(
        args.model, device,
        text_encoders_cpu=args.text_encoders_cpu,
        vae_tiling=args.vae_tiling,
    )
    print("Loading CLIP predictor...")
    predictor = CLIPPromptPredictor(device=device, torch_dtype=torch.float32)

    common_kwargs = dict(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    # Cell A first — no scheduler patch, baseline measurement.
    res_a = None
    if "A" in args.cells:
        res_a = run_cell(
            pipe, predictor, prompts,
            output_dir=output_dir / "cell_a",
            cell_name="A",
            use_tfg=False,
            **common_kwargs,
        )
        with open(output_dir / "cell_a_results.json", "w") as f:
            json.dump(res_a, f, indent=2)
        print(f"Cell A: mean CLIP = {res_a['mean_clip_score']:.4f}  ({res_a['wall_time_s']:.0f}s)")
    else:
        # Reuse previously saved Cell A results so the summary still has both cells.
        a_path = output_dir / "cell_a_results.json"
        if a_path.exists():
            with open(a_path) as f:
                res_a = json.load(f)
            print(f"Cell A: reusing saved mean CLIP = {res_a['mean_clip_score']:.4f}")

    # Cell B: patch scheduler, generate with TFG.
    res_b = None
    if "B" in args.cells:
        res_b = run_cell(
            pipe, predictor, prompts,
            output_dir=output_dir / "cell_b",
            cell_name="B",
            use_tfg=True,
            tfg_scale=args.tfg_scale,
            tfg_every_n=args.tfg_every_n,
            **common_kwargs,
        )
        with open(output_dir / "cell_b_results.json", "w") as f:
            json.dump(res_b, f, indent=2)
        print(f"Cell B: mean CLIP = {res_b['mean_clip_score']:.4f}  ({res_b['wall_time_s']:.0f}s)")

    if res_a is not None and res_b is not None:
        delta = res_b["mean_clip_score"] - res_a["mean_clip_score"]
        summary = {
            "cell_a_mean_clip": res_a["mean_clip_score"],
            "cell_b_mean_clip": res_b["mean_clip_score"],
            "delta_b_minus_a": delta,
            "cell_a_wall_s": res_a["wall_time_s"],
            "cell_b_wall_s": res_b["wall_time_s"],
            "per_prompt_paired": [
                {
                    "id": a["id"],
                    "prompt": a["prompt"],
                    "category": a.get("category"),
                    "clip_a": a["clip_score"],
                    "clip_b": b["clip_score"],
                    "delta": b["clip_score"] - a["clip_score"],
                }
                for a, b in zip(res_a["items"], res_b["items"])
            ],
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        wins = sum(1 for p in summary["per_prompt_paired"] if p["delta"] > 0)
        print(f"\n=== SMOKE SUMMARY ===")
        print(f"Cell A (no TFG)  mean CLIP: {res_a['mean_clip_score']:.4f}")
        print(f"Cell B (with TFG) mean CLIP: {res_b['mean_clip_score']:.4f}")
        print(f"Delta (B - A): {delta:+.4f}")
        print(f"TFG wins per prompt: {wins}/{len(prompts)}")


if __name__ == "__main__":
    main()
