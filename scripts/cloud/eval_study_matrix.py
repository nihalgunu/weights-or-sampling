#!/usr/bin/env python3
"""
TFG-vs-GRPO study: full eval matrix for ONE reward on ONE prompt set.

Arms (all scored on the same scorers, paired seeds across arms):
  - no_lora                  base model
  - tfg_rho{R}               TFG on base, guided by the SAME reward model the
                             RL trained on (CLIP-L or PickScore), R in --tfg-rhos
  - {checkpoint-N}           FlowGRPO group=8 LoRA endpoints
  - stack_{ckpt}_rho{R}      TFG on top of the best RL checkpoint, R in --stack-rhos

Every image is scored with ALL of: CLIP-L (logits/30), PickScore (logit/26),
jpeg-compressibility. The primary reward is whichever the arm optimizes; the
other two are cross-metrics (reward-hacking / off-manifold probes: a policy or
guidance that wins its own reward by blurring shows up as a jpeg spike and a
cross-reward drop).

Seeds: --seed-bases "0,1000,2000" -> for prompt i, seeds b+i. Base 0 reproduces
the seeds of every prior flowgrpo_*_eval.json run (paired comparability).

Usage (on the box):
    python3 scripts/cloud/eval_study_matrix.py \
        --reward clip \
        --prompts-file outputs/eval_prompts_ddrl.txt \
        --ckpt-dir outputs/checkpoints/flowgrpo_clip \
        --tfg-rhos 2,5,10,20 --stack-rhos 20 \
        --seed-bases 0,1000,2000 \
        --image-dir outputs/study_clip_images \
        --out outputs/study_clip_matrix.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.clip_predictor import CLIPPromptPredictor
from scripts.pickscore_predictor import PickScorePromptPredictor
from scripts.tfg_flow import (
    PromptContext,
    TFGFlowGuidance,
    patch_scheduler_step,
    unpatch_scheduler_step,
)
from scripts.cloud.run_smoke_cells import build_pipe, encode_prompt_offloaded
from scripts.cloud.eval_lora_compressibility import (
    ClipRewardScorer,
    PickScoreScorer,
    jpeg_compressibility,
)


def load_prompt_lines(path: Path, n: int) -> list[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        lines = [d["prompt"] for d in data]
    else:
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return lines[:n]


def pick_rl_endpoints(ckpt_dir: Path, target: int = 300) -> list[str]:
    """Same policy as prior evals: checkpoint closest to `target` grad updates
    (matched-DDRL-compute / observed peak) + the final checkpoint."""
    if not ckpt_dir.is_dir():
        return []
    ckpts = sorted(
        [d.name for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")],
        key=lambda nm: int(re.search(r"\d+", nm).group()),
    )
    if not ckpts:
        return []
    nums = [int(re.search(r"\d+", nm).group()) for nm in ckpts]
    mid = min(range(len(nums)), key=lambda i: abs(nums[i] - target))
    picks = []
    for i in (mid, len(ckpts) - 1):
        if ckpts[i] not in picks:
            picks.append(ckpts[i])
    return picks


def reset_transformer(pipe, base_state_dict):
    if hasattr(pipe.transformer, "unload"):
        pipe.transformer = pipe.transformer.unload()
    pipe.transformer.load_state_dict(base_state_dict)


def load_lora(pipe, lora_path: Path):
    from peft import PeftModel

    assert lora_path.exists(), f"missing {lora_path}"
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer, str(lora_path), is_trainable=False
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--reward", choices=["clip", "pickscore", "ensemble", "aesthetic", "imagereward"], required=True,
                    help="Primary reward: selects the TFG guidance predictor and the headline scorer.")
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--num-prompts", type=int, default=30)
    ap.add_argument("--num-steps", type=int, default=40)
    ap.add_argument("--guidance-scale", type=float, default=4.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seed-bases", default="0,1000,2000")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--rl-endpoints", nargs="*", default=None,
                    help="Explicit checkpoint names; default: auto (closest-to-300 + last).")
    ap.add_argument("--rl-lora", action="append", default=[],
                    help="Explicit RL arm as NAME=PATH (repeatable). PATH is a LoRA dir. "
                         "When given, replaces the ckpt-dir auto-discovery for RL arms.")
    ap.add_argument("--stack", action="append", default=[],
                    help="Explicit stacked arm as NAME=PATH:RHO (repeatable). "
                         "When given, replaces the default best-ckpt stacking.")
    ap.add_argument("--tfg-rhos", default="2,5,10,20")
    ap.add_argument("--tfg-mus", default="",
                    help="mu-only TFG arms (x0-refinement branch), comma list.")
    ap.add_argument("--tfg-combos", default="",
                    help="rho:mu combined TFG arms, comma list (e.g. 20:20).")
    ap.add_argument("--stack-rhos", default="20",
                    help="TFG rhos to run on top of the best RL checkpoint. Empty string to skip.")
    ap.add_argument("--tfg-apply-every-n", type=int, default=1)
    ap.add_argument("--image-dir", default=None, help="If set, save every PNG under <dir>/<arm>/<seedbase>/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda"
    t_start = time.time()

    prompts = load_prompt_lines(Path(args.prompts_file), args.num_prompts)
    seed_bases = [int(s) for s in args.seed_bases.split(",") if s != ""]
    tfg_rhos = [float(x) for x in args.tfg_rhos.split(",") if x != ""]
    tfg_mus = [float(x) for x in args.tfg_mus.split(",") if x != ""]
    tfg_combos = [tuple(float(v) for v in c.split(":")) for c in args.tfg_combos.split(",") if c != ""]
    stack_rhos = [float(x) for x in args.stack_rhos.split(",") if x != ""]

    ckpt_dir = Path(args.ckpt_dir)

    # --- Arm list ---
    arms: list[dict] = [{"name": "no_lora", "lora": None, "rho": None, "mu": None}]
    for r in tfg_rhos:
        arms.append({"name": f"tfg_rho{r:g}", "lora": None, "rho": r, "mu": None})
    for m in tfg_mus:
        arms.append({"name": f"tfg_mu{m:g}", "lora": None, "rho": None, "mu": m})
    for r, m in tfg_combos:
        arms.append({"name": f"tfg_rho{r:g}_mu{m:g}", "lora": None, "rho": r, "mu": m})

    if args.rl_lora or args.stack:
        # Explicit multi-seed mode (pub-grade study): arms given on the CLI.
        rl_eps = []
        for spec in args.rl_lora:
            name, path = spec.split("=", 1)
            rl_eps.append(name)
            arms.append({"name": name, "lora": Path(path), "rho": None, "mu": None})
        for spec in args.stack:
            name, rest = spec.split("=", 1)
            parts = rest.split(":")
            if len(parts) >= 3:
                path, rho, mu = ":".join(parts[:-2]), float(parts[-2]), float(parts[-1])
            else:
                path, rho, mu = ":".join(parts[:-1]), float(parts[-1]), None
            arms.append({"name": name, "lora": Path(path), "rho": rho, "mu": mu})
        best_ckpt = rl_eps[0] if rl_eps else None
        print(f"[matrix] explicit arms: rl={rl_eps} stack={[s.split('=')[0] for s in args.stack]}")
    else:
        rl_eps = args.rl_endpoints if args.rl_endpoints else pick_rl_endpoints(ckpt_dir)
        best_ckpt = rl_eps[0] if rl_eps else None
        print(f"[matrix] RL endpoints: {rl_eps or 'NONE'}; stacked on: {best_ckpt}")
        for ep in rl_eps:
            arms.append({"name": ep, "lora": ckpt_dir / ep / "lora", "rho": None, "mu": None})
        if best_ckpt is not None:
            for r in stack_rhos:
                arms.append({"name": f"stack_{best_ckpt}_rho{r:g}",
                             "lora": ckpt_dir / best_ckpt / "lora", "rho": r})

    # --- Models ---
    print(f"[matrix] loading {args.base} (bf16, text encoders offloaded)")
    pipe = build_pipe(args.base, device, text_encoders_cpu=True, vae_tiling=True)
    pipe.set_progress_bar_config(disable=True)
    base_state_dict = {k: v.detach().clone() for k, v in pipe.transformer.state_dict().items()}

    if args.reward == "clip":
        predictor = CLIPPromptPredictor(device=device)
    elif args.reward == "ensemble":
        from scripts.ensemble_predictor import EnsemblePromptPredictor
        predictor = EnsemblePromptPredictor(device=device)
    elif args.reward == "aesthetic":
        from scripts.aesthetic_predictor import AestheticPredictor
        predictor = AestheticPredictor(device=device)
    elif args.reward == "imagereward":
        from scripts.imagereward_predictor import ImageRewardPredictor
        predictor = ImageRewardPredictor(device=device)
    else:
        predictor = PickScorePromptPredictor(device=device)
    # For "ensemble", the per-image ensemble score is recomputable offline as
    # 0.5*clip + 0.5*pickscore from the stored per-metric columns; use clip
    # for progress display.
    display = "clip" if args.reward in ("ensemble", "aesthetic", "imagereward") else args.reward

    scorers = {
        "clip": ClipRewardScorer(device=device),
        "pickscore": PickScoreScorer(device=device),
    }
    if args.reward in ("aesthetic", "imagereward"):
        import torch as _t
        import torchvision.transforms.functional as _TF
        _pred = predictor
        def _pil_score(img, prompt):
            x = _TF.to_tensor(img).to(device) * 2 - 1
            with _t.no_grad():
                return float(_pred(x.unsqueeze(0), [prompt]).item())
        scorers[args.reward] = _pil_score

    # Prompt-embedding cache (embeds are arm/seed independent).
    embed_cache: dict[str, tuple] = {}

    def get_embeds(prompt: str):
        if prompt not in embed_cache:
            embed_cache[prompt] = encode_prompt_offloaded(pipe, prompt, device)
        return embed_cache[prompt]

    # Pre-encode all prompts once (T5 shuttles once per prompt, not per arm).
    for p in prompts:
        get_embeds(p)
    print(f"[matrix] encoded {len(prompts)} prompts in {time.time()-t_start:.0f}s")

    results = {
        "meta": {
            "base": args.base, "reward": args.reward,
            "num_steps": args.num_steps, "guidance_scale": args.guidance_scale,
            "num_prompts": len(prompts), "prompts": prompts,
            "seed_bases": seed_bases,
            "tfg": {"apply_every_n": args.tfg_apply_every_n,
                    "apply_range": [0.1, 0.9], "recurrence_steps": 1,
                    "predictor": "clip_l_cosine" if args.reward == "clip" else "pickscore_cosine"},
            "arms": [a["name"] for a in arms],
            "rl_endpoints": rl_eps, "stacked_on": best_ckpt,
        },
        "scores": {},   # arm -> scorer -> {seed_base -> [per-prompt floats]}
        "timing": {},
    }

    for arm in arms:
        name = arm["name"]
        print(f"\n[matrix] === arm: {name} ===  ({time.time()-t_start:.0f}s elapsed)")
        reset_transformer(pipe, base_state_dict)
        if arm["lora"] is not None:
            load_lora(pipe, Path(arm["lora"]))
        pipe.transformer.to(device, dtype=torch.bfloat16)

        guidance = None
        if arm["rho"] is not None or arm.get("mu") is not None:
            guidance = TFGFlowGuidance(
                predictor=predictor,
                guidance_scale=arm["rho"] or 0.0,
                mu=arm.get("mu") or 0.0,
                recurrence_steps=1,
                apply_every_n_steps=args.tfg_apply_every_n,
                apply_range=(0.1, 0.9),
            )
            patch_scheduler_step(pipe, guidance)

        arm_keys = ["clip", "pickscore", "jpeg"]
        if args.reward in ("aesthetic", "imagereward"):
            arm_keys.append(args.reward)
        arm_scores = {k: {} for k in arm_keys}
        t_arm = time.time()
        try:
            for b in seed_bases:
                for k in arm_scores:
                    arm_scores[k][str(b)] = []
                for i, prompt in enumerate(prompts):
                    gen = torch.Generator(device=device).manual_seed(b + i)
                    pos, neg, pos_p, neg_p = get_embeds(prompt)
                    if guidance is not None:
                        guidance.set_prompt_context(PromptContext(
                            prompt_embeds=pos, pooled_prompt_embeds=pos_p,
                            prompt_text=[prompt],
                        ))
                    with torch.no_grad():
                        img = pipe(
                            prompt_embeds=pos, negative_prompt_embeds=neg,
                            pooled_prompt_embeds=pos_p, negative_pooled_prompt_embeds=neg_p,
                            num_inference_steps=args.num_steps,
                            guidance_scale=args.guidance_scale,
                            height=args.height, width=args.width,
                            generator=gen,
                        ).images[0]
                    arm_scores["clip"][str(b)].append(scorers["clip"](img, prompt))
                    arm_scores["pickscore"][str(b)].append(scorers["pickscore"](img, prompt))
                    arm_scores["jpeg"][str(b)].append(jpeg_compressibility(img))
                    if args.reward in ("aesthetic", "imagereward"):
                        arm_scores[args.reward][str(b)].append(scorers[args.reward](img, prompt))
                    if args.image_dir:
                        d = Path(args.image_dir) / name / str(b)
                        d.mkdir(parents=True, exist_ok=True)
                        img.save(d / f"{i:03d}.png")
                    if i == 0 or i == len(prompts) - 1:
                        print(f"  seed_base={b} [{i+1}/{len(prompts)}] "
                              f"{display}={arm_scores[display][str(b)][-1]:+.4f} "
                              f"({time.time()-t_arm:.0f}s)")
        finally:
            if guidance is not None:
                unpatch_scheduler_step(pipe)

        results["scores"][name] = arm_scores
        results["timing"][name] = time.time() - t_arm
        prim = [v for b in seed_bases for v in arm_scores[display][str(b)]]
        print(f"  -> {name}: {display} mean={statistics.mean(prim):+.4f} "
              f"n={len(prim)} ({time.time()-t_arm:.0f}s)")

        # Checkpoint the JSON after every arm so a crash loses nothing.
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1))

    # --- Summary: paired deltas vs no_lora on every scorer ---
    summary = {}
    base_scores = results["scores"]["no_lora"]
    for name, sc in results["scores"].items():
        s_entry = {}
        for metric in ["clip", "pickscore", "jpeg"]:
            flat = [v for b in seed_bases for v in sc[metric][str(b)]]
            e = {"mean": statistics.mean(flat), "std": statistics.stdev(flat)}
            if name != "no_lora":
                deltas = [a - c for b in seed_bases
                          for a, c in zip(sc[metric][str(b)], base_scores[metric][str(b)])]
                n = len(deltas)
                dm = statistics.mean(deltas)
                dsd = statistics.stdev(deltas)
                e.update({
                    "delta_vs_no_lora": dm,
                    "delta_sd": dsd,
                    "delta_se": dsd / n ** 0.5,
                    "z": dm / (dsd / n ** 0.5) if dsd > 0 else 0.0,
                    "wins": sum(1 for d in deltas if d > 0),
                    "n": n,
                })
            s_entry[metric] = e
        summary[name] = s_entry
    results["summary"] = summary

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\n[matrix] wrote {args.out} ({time.time()-t_start:.0f}s total)")
    print(f"\n=== SUMMARY (primary metric: {args.reward}, paired vs no_lora) ===")
    for name, s in summary.items():
        e = s[display]
        line = f"{name:32s} mean={e['mean']:+.4f}"
        if "delta_vs_no_lora" in e:
            line += (f"  d={e['delta_vs_no_lora']:+.4f} se={e['delta_se']:.4f} "
                     f"z={e['z']:+.1f} wins={e['wins']}/{e['n']}")
        print(line)


if __name__ == "__main__":
    main()
