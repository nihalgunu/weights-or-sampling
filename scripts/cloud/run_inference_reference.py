#!/usr/bin/env python3
"""
Cell A_ref — SD3.5-M via HF InferenceClient (fal-ai provider) as cross-validation.

Why this exists:
    Our Cell A uses a local diffusers StableDiffusion3Pipeline. If we misconfigured
    the pipeline (wrong scheduler, wrong VAE scaling, wrong negative prompt, wrong
    CFG) then Cell A could look weird and Cell B (TFG) would be measured against
    a biased baseline. Running the same 30 prompts through HF's hosted endpoint
    — which uses StabilityAI's reference implementation — and scoring with the
    SAME CLIPPromptPredictor gives us an independent number for "what SD3.5-M
    looks like on these prompts."

    If Cell A and Cell A_ref agree (within noise), our local setup is sane.
    If they diverge, something is wrong in Cell A and we need to fix before
    trusting any Cell B result.

Caveats to write into result.md:
    - fal-ai may use a different number of inference steps / CFG scale than ours.
      We force num_inference_steps=28 and guidance_scale=4.5 to match if the
      provider honors those args (not all providers do).
    - fal-ai charges per image (~$0.01-0.03 at the time of writing). 30 prompts
      is well under a dollar.
    - Score is computed with our local predictor for apples-to-apples.
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


def load_prompts(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def score_image(predictor: CLIPPromptPredictor, pil_image: Image.Image, prompt: str) -> float:
    import torchvision.transforms.functional as TF

    img = TF.to_tensor(pil_image).to(predictor.device, dtype=predictor.dtype)
    img = img * 2.0 - 1.0
    img = img.unsqueeze(0)
    with torch.no_grad():
        s = predictor(img, [prompt])
    return float(s.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", default="scripts/cloud/prompts_smoke.json")
    parser.add_argument("--output_dir", default="outputs/smoke/cell_a_ref")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--provider", default="fal-ai",
                        help="HF InferenceClient provider. fal-ai is the default "
                             "because it supports SD3.5-M; the HF-managed endpoint "
                             "has spotty coverage for gated models.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import os
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: set HF_TOKEN in env", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import InferenceClient

    client = InferenceClient(provider=args.provider, api_key=hf_token)
    prompts = load_prompts(Path(args.prompts_file))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = CLIPPromptPredictor(device=device, torch_dtype=torch.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"cell": "A_ref", "provider": args.provider, "model": args.model, "items": []}
    t0 = time.time()
    for p in tqdm(prompts, desc="A_ref (fal-ai)"):
        prompt_text = p["prompt"]

        # fal-ai accepts num_inference_steps and guidance_scale as extra kwargs,
        # but keys/accepted values vary. We pass both common names; unknown
        # kwargs are usually ignored rather than errored.
        for attempt in range(3):
            try:
                image = client.text_to_image(
                    prompt_text,
                    model=args.model,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed + p["id"],
                )
                break
            except Exception as e:
                if attempt == 2:
                    print(f"[warn] prompt {p['id']} failed 3x: {e}")
                    image = None
                    break
                time.sleep(2 ** attempt)

        if image is None:
            results["items"].append({
                "id": p["id"], "prompt": prompt_text,
                "category": p.get("category"),
                "clip_score": None, "image_path": None, "error": "fetch failed",
            })
            continue

        image_path = output_dir / f"{p['id']:03d}.png"
        image.save(image_path)
        score = score_image(predictor, image, prompt_text)
        results["items"].append({
            "id": p["id"], "prompt": prompt_text,
            "category": p.get("category"),
            "image_path": str(image_path),
            "clip_score": score,
        })

    total = time.time() - t0
    results["wall_time_s"] = total
    valid = [i for i in results["items"] if i.get("clip_score") is not None]
    if valid:
        results["mean_clip_score"] = sum(i["clip_score"] for i in valid) / len(valid)
    else:
        results["mean_clip_score"] = None

    with open(Path(args.output_dir).parent / "cell_a_ref_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nA_ref: mean CLIP = {results['mean_clip_score']} on {len(valid)}/{len(prompts)} images ({total:.0f}s)")


if __name__ == "__main__":
    main()
