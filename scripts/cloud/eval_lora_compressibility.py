#!/usr/bin/env python3
"""
Offline eval: compare LoRA endpoints on a chosen reward (jpeg or clip).

For each endpoint (e.g. no-LoRA + checkpoints), generates one image per prompt
with paired seeds (same seed across endpoints — variance reduction), scores
with the chosen reward, and dumps per-prompt rewards plus summary stats.

Usage (on a 1xA100 box, after rsync):
    python3 scripts/cloud/eval_lora_compressibility.py \\
        --base stabilityai/stable-diffusion-3.5-medium \\
        --ckpt-dir outputs/checkpoints/flowgrpo \\
        --prompts-file repos/flow_grpo/dataset/ocr/test.txt \\
        --num-prompts 30 --num-steps 40 \\
        --reward jpeg \\
        --out outputs/flowgrpo_eval.json
"""
import argparse
import json
import os
import statistics
import time
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel


def jpeg_compressibility(img: Image.Image) -> float:
    """Match flow_grpo/rewards.py:22 — neg JPEG bytes / 500. Higher = more compressible."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    size_kb = buf.tell() / 1000
    return -size_kb / 500


class ClipRewardScorer:
    """CLIP cosine similarity; matches flow_grpo's clip_scorer.ClipScorer.

    Uses logits_per_image / 30 to mirror what FlowGRPO trains against, so train
    and eval reward are on the same scale. Higher = better text-image alignment.
    """

    def __init__(self, device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor
        import torchvision.transforms as T

        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        cfg = self.processor.image_processor.to_dict()
        size = cfg.get("size")
        if isinstance(size, dict):
            size = size.get("shortest_edge") or (size["height"], size["width"])
        crop = cfg.get("crop_size")
        if isinstance(crop, dict):
            crop = (crop["height"], crop["width"])
        self.tform = T.Compose([
            T.Resize(size if isinstance(size, tuple) else (size, size)),
            T.CenterCrop(crop if isinstance(crop, tuple) else (crop, crop)),
            T.Normalize(mean=self.processor.image_processor.image_mean,
                        std=self.processor.image_processor.image_std),
        ])

    @torch.no_grad()
    def __call__(self, img: Image.Image, prompt: str) -> float:
        import numpy as np
        arr = np.array(img.convert("RGB"))
        pixels = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        pixels = self.tform(pixels).to(self.device)
        text = self.processor(text=[prompt], padding="max_length", truncation=True,
                              return_tensors="pt").to(self.device)
        out = self.model(pixel_values=pixels, **text)
        return float(out.logits_per_image.diagonal().item() / 30.0)


class PickScoreScorer:
    """PickScore (yuvalkirstain/PickScore_v1) — same scoring flow_grpo trains
    against (rewards.py `pickscore` reward). Output is `logit_scale * sim / 26`,
    typically in ~0.0-1.0 range; higher = more human-preferred.
    """

    def __init__(self, device: str = "cuda"):
        from transformers import CLIPProcessor, CLIPModel
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = CLIPModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device).eval()

    @torch.no_grad()
    def __call__(self, img: Image.Image, prompt: str) -> float:
        img_in = self.processor(images=[img], padding=True, truncation=True,
                                max_length=77, return_tensors="pt")
        img_in = {k: v.to(self.device) for k, v in img_in.items()}
        txt_in = self.processor(text=[prompt], padding=True, truncation=True,
                                max_length=77, return_tensors="pt")
        txt_in = {k: v.to(self.device) for k, v in txt_in.items()}
        ie = self.model.get_image_features(**img_in)
        ie = ie / ie.norm(p=2, dim=-1, keepdim=True)
        te = self.model.get_text_features(**txt_in)
        te = te / te.norm(p=2, dim=-1, keepdim=True)
        score = (self.model.logit_scale.exp() * (te @ ie.T)).diag() / 26.0
        return float(score.item())


def load_prompts(path: Path, n: int) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return lines[:n]


def reset_transformer(pipe, base_state_dict):
    """Strip any LoRA wrapper and restore the base transformer's weights.

    get_base_model() returns the wrapped reference but the LoRA layers are
    structurally injected, so state_dict keys still differ. unload() removes
    the LoRA layers and returns the bare base.
    """
    if hasattr(pipe.transformer, "unload"):
        pipe.transformer = pipe.transformer.unload()
    pipe.transformer.load_state_dict(base_state_dict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--ckpt-dir", required=True, help="dir containing checkpoint-N/lora subdirs")
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--num-prompts", type=int, default=30)
    ap.add_argument("--num-steps", type=int, default=40)
    ap.add_argument("--guidance-scale", type=float, default=4.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reward", choices=["jpeg", "clip", "pickscore"], default="jpeg",
                    help="jpeg: jpeg_compressibility. clip: CLIP-L logits/30. pickscore: PickScore_v1 logits/26.")
    ap.add_argument("--endpoints", nargs="+",
                    default=["no_lora", "checkpoint-2", "checkpoint-180", "checkpoint-374"])
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    score_fn = jpeg_compressibility
    if args.reward == "clip":
        score_fn = ClipRewardScorer(device=device)
    elif args.reward == "pickscore":
        score_fn = PickScoreScorer(device=device)

    print(f"[{time.strftime('%H:%M:%S')}] loading {args.base} (bf16)")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.base, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    base_state_dict = {k: v.detach().clone() for k, v in pipe.transformer.state_dict().items()}

    prompts = load_prompts(Path(args.prompts_file), args.num_prompts)
    seeds = list(range(args.num_prompts))
    print(f"[{time.strftime('%H:%M:%S')}] {len(prompts)} prompts × {len(args.endpoints)} endpoints")

    results = {"meta": {
        "base": args.base, "num_steps": args.num_steps,
        "num_prompts": len(prompts), "guidance_scale": args.guidance_scale,
        "reward": args.reward,
        "prompts": prompts, "seeds": seeds,
    }, "rewards": {}}

    for ep in args.endpoints:
        print(f"\n[{time.strftime('%H:%M:%S')}] === endpoint: {ep} ===")
        reset_transformer(pipe, base_state_dict)
        if ep != "no_lora":
            lora_path = Path(args.ckpt_dir) / ep / "lora"
            assert lora_path.exists(), f"missing {lora_path}"
            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(lora_path))
            pipe.transformer.set_adapter("default")
        pipe.transformer.to(device, dtype=dtype)

        rewards = []
        t0 = time.time()
        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            gen = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                img = pipe(
                    prompt,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    generator=gen,
                ).images[0]
            r = score_fn(img) if args.reward == "jpeg" else score_fn(img, prompt)
            rewards.append(r)
            if i < 3 or i == len(prompts) - 1:
                print(f"  [{i+1}/{len(prompts)}] r={r:+.4f}  ({time.time()-t0:.0f}s elapsed)")
        results["rewards"][ep] = rewards
        m, s = statistics.mean(rewards), statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        print(f"  → mean={m:+.4f}  std={s:.4f}  ({time.time()-t0:.0f}s)")

    # Summary block: paired deltas vs no_lora.
    summary = {}
    for ep, rs in results["rewards"].items():
        summary[ep] = {"mean": statistics.mean(rs), "std": statistics.stdev(rs) if len(rs) > 1 else 0.0}
    if "no_lora" in results["rewards"]:
        base_rs = results["rewards"]["no_lora"]
        for ep, rs in results["rewards"].items():
            if ep == "no_lora":
                continue
            deltas = [a - b for a, b in zip(rs, base_rs)]
            summary[ep]["delta_vs_no_lora"] = statistics.mean(deltas)
            summary[ep]["delta_std"] = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    results["summary"] = summary

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[{time.strftime('%H:%M:%S')}] wrote {out}")
    print("\n=== SUMMARY ===")
    for ep, s in summary.items():
        line = f"{ep:20s}  mean={s['mean']:+.4f}  std={s['std']:.4f}"
        if "delta_vs_no_lora" in s:
            line += f"  Δ_vs_no_lora={s['delta_vs_no_lora']:+.4f} (std {s['delta_std']:.4f})"
        print(line)


if __name__ == "__main__":
    main()
