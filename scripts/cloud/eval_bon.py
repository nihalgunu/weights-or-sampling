#!/usr/bin/env python3
"""
Best-of-N baseline at matched inference compute.

TFG costs ~3.9x base inference, so the standard test-time alternative is:
generate N=4 candidates per prompt and keep the one the reward model scores
highest (reward reranking — the same information TFG uses, spent differently).

Arms: n1_<policy> (single sample, candidate 0) and bon<N>_<policy>, for
policy in {no_lora} + any --rl-lora NAME=PATH. Candidate 0 uses the study's
standard seed (b+i) so n1 arms pair exactly with prior matrices; candidates
k>0 use offset seeds. The selected BoN image is also cross-scored
(clip/pickscore/jpeg) for hacking probes.

Usage (on the box):
    python3 scripts/cloud/eval_bon.py --reward clip \
        --prompts-file outputs/eval_prompts_pub.txt --num-prompts 150 \
        --n 4 --seed-bases 0,1000,2000 \
        --rl-lora rl_s42=outputs/checkpoints/flowgrpo_clip_pub_s42/checkpoint-262/lora \
        --out outputs/study_bon_clip_matrix.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cloud.run_smoke_cells import build_pipe, encode_prompt_offloaded
from scripts.cloud.eval_lora_compressibility import (
    ClipRewardScorer, PickScoreScorer, jpeg_compressibility,
)
from scripts.cloud.eval_study_matrix import reset_transformer, load_lora, load_prompt_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--reward", choices=["clip", "pickscore"], required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--num-prompts", type=int, default=150)
    ap.add_argument("--num-steps", type=int, default=40)
    ap.add_argument("--guidance-scale", type=float, default=4.5)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed-bases", default="0,1000,2000")
    ap.add_argument("--rl-lora", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda"
    t0 = time.time()
    prompts = load_prompt_lines(Path(args.prompts_file), args.num_prompts)
    seed_bases = [int(s) for s in args.seed_bases.split(",") if s != ""]

    pipe = build_pipe(args.base, device, text_encoders_cpu=True, vae_tiling=True)
    pipe.set_progress_bar_config(disable=True)
    base_state_dict = {k: v.detach().clone() for k, v in pipe.transformer.state_dict().items()}

    scorers = {"clip": ClipRewardScorer(device=device), "pickscore": PickScoreScorer(device=device)}
    prim = args.reward

    embed_cache = {}
    def get_embeds(p):
        if p not in embed_cache:
            embed_cache[p] = encode_prompt_offloaded(pipe, p, device)
        return embed_cache[p]
    for p in prompts:
        get_embeds(p)
    print(f"[bon] encoded {len(prompts)} prompts ({time.time()-t0:.0f}s)")

    policies = [("no_lora", None)]
    for spec in args.rl_lora:
        name, path = spec.split("=", 1)
        policies.append((name, Path(path)))

    results = {"meta": {"base": args.base, "reward": prim, "n": args.n,
                        "num_prompts": len(prompts), "seed_bases": seed_bases,
                        "prompts": prompts,
                        "arms": [f"{k}_{nm}" for nm, _ in policies for k in ("n1", f"bon{args.n}")]},
               "scores": {}, "cand_scores": {}}

    for pol_name, lora in policies:
        reset_transformer(pipe, base_state_dict)
        if lora is not None:
            load_lora(pipe, lora)
        pipe.transformer.to(device, dtype=torch.bfloat16)
        cand_all = {str(b): [] for b in seed_bases}
        n1 = {m: {str(b): [] for b in seed_bases} for m in ["clip", "pickscore", "jpeg"]}
        bon = {m: {str(b): [] for b in seed_bases} for m in ["clip", "pickscore", "jpeg"]}
        t_arm = time.time()
        for b in seed_bases:
            for i, prompt in enumerate(prompts):
                pos, neg, pos_p, neg_p = get_embeds(prompt)
                cand_imgs, cand_scores = [], []
                for k in range(args.n):
                    seed = b + i if k == 0 else b + i + 7777777 * k
                    gen = torch.Generator(device=device).manual_seed(seed)
                    with torch.no_grad():
                        img = pipe(prompt_embeds=pos, negative_prompt_embeds=neg,
                                   pooled_prompt_embeds=pos_p, negative_pooled_prompt_embeds=neg_p,
                                   num_inference_steps=args.num_steps,
                                   guidance_scale=args.guidance_scale,
                                   height=512, width=512, generator=gen).images[0]
                    cand_imgs.append(img)
                    cand_scores.append(scorers[prim](img, prompt))
                cand_all[str(b)].append(cand_scores)
                best = max(range(args.n), key=lambda k: cand_scores[k])
                for m in ["clip", "pickscore"]:
                    n1[m][str(b)].append(scorers[m](cand_imgs[0], prompt))
                    bon[m][str(b)].append(scorers[m](cand_imgs[best], prompt))
                n1["jpeg"][str(b)].append(jpeg_compressibility(cand_imgs[0]))
                bon["jpeg"][str(b)].append(jpeg_compressibility(cand_imgs[best]))
                if i == 0 or i == len(prompts) - 1:
                    print(f"[bon] {pol_name} b={b} [{i+1}/{len(prompts)}] "
                          f"n1={n1[prim][str(b)][-1]:+.4f} bon={bon[prim][str(b)][-1]:+.4f} "
                          f"({time.time()-t_arm:.0f}s)")
        results["cand_scores"][pol_name] = cand_all
        results["scores"][f"n1_{pol_name}"] = n1
        results["scores"][f"bon{args.n}_{pol_name}"] = bon
        Path(args.out).write_text(json.dumps(results, indent=1))
        flat = [v for b in seed_bases for v in bon[prim][str(b)]]
        print(f"[bon] {pol_name}: bon{args.n} mean={statistics.mean(flat):+.4f} ({time.time()-t_arm:.0f}s)")

    print(f"[bon] wrote {args.out} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
