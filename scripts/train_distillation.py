#!/usr/bin/env python3
"""
Stage 2 of TFG Distillation: supervised flow-matching LoRA finetune against
teacher latents from a Stage-1 manifest.

Loss is the standard SD3 flow-matching objective:
    given teacher latent x0_T and a random sigma, form
      x_t = sigma * noise + (1 - sigma) * x0_T
    policy predicts velocity v_pred from (x_t, t, prompt_embeds); target is
      v_target = noise - x0_T
    loss = MSE(v_pred, v_target).

This is ordinary supervised finetuning with flow-matching noising. No rollout,
no reward, no ref model. Vastly lower variance than RL — exactly why we
switched after mini-DDRL didn't train. A single optimizer step costs one
transformer forward + backward on a single (latent, prompt) pair, ~1-3s on A10.

Memory: policy transformer + LoRA + text encoders (offloaded per call). No ref
model. Easily fits on 24 GB.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_ddrl import (
    _encode_prompts_cfg,
    setup_policy_model,
)


def load_manifest(path: str) -> tuple[dict, list]:
    with open(path) as f:
        manifest = json.load(f)
    items = manifest["items"]
    return manifest, items


def compute_fm_loss(
    transformer,
    teacher_latent: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    scheduler,
    device: torch.device,
) -> torch.Tensor:
    """One flow-matching training step for a single (teacher_latent, prompt).

    Samples sigma ~ Uniform(0, 1) after the scheduler's dynamic shift is
    applied. A truly faithful reproduction of SD3's training objective uses
    a logit-normal sigma distribution (Esser et al.); uniform is close enough
    for a LoRA finetune on a small dataset and saves a dependency on the
    scheduler's sigma sampler. We keep it uniform and document the deviation.
    """
    B = teacher_latent.shape[0]
    sigma = torch.rand(B, device=device, dtype=torch.float32).clamp(min=0.05, max=0.95)
    noise = torch.randn_like(teacher_latent)
    sig_exp = sigma.view(-1, 1, 1, 1)
    x_t = sig_exp * noise + (1.0 - sig_exp) * teacher_latent
    v_target = (noise - teacher_latent).to(torch.float32)

    num_train = int(scheduler.config.num_train_timesteps)
    t_batch = (sigma * num_train).to(device)

    v_pred = transformer(
        hidden_states=x_t.to(transformer.dtype),
        timestep=t_batch,
        encoder_hidden_states=prompt_embeds.to(transformer.dtype),
        pooled_projections=pooled_prompt_embeds.to(transformer.dtype),
        return_dict=False,
    )[0].to(torch.float32)

    return F.mse_loss(v_pred, v_target)


def main():
    parser = argparse.ArgumentParser(
        description="TFG distillation: flow-matching LoRA finetune on teacher latents"
    )
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to Stage-1 manifest.json with teacher latents.")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/distilled")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of (teacher_latent, prompt) pairs per step.")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true",
                        help="Park T5-XXL on CPU, shuttle per encode. Required on 24GB.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = yaml.safe_load(open(args.config))
    model_name = config.get("model", {}).get("name", "stabilityai/stable-diffusion-3.5-medium")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load policy model + LoRA (same helper as DDRL training path).
    policy_pipe = setup_policy_model(
        model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=device,
        t5_cpu=args.t5_cpu,
    )
    policy_pipe.transformer.train()

    # Load manifest + teacher latents into RAM. 200 samples × (16*64*64*4 bytes)
    # = ~50 MB total, fine to hold in memory.
    manifest, items = load_manifest(args.manifest)
    print(f"loaded manifest with {len(items)} teacher samples (ρ={manifest.get('tfg_rho')})")
    latents_by_id = {}
    manifest_dir = Path(args.manifest).parent
    for item in items:
        lat = torch.load(manifest_dir / item["latent_path"], map_location="cpu")
        if lat.dim() == 3:
            lat = lat.unsqueeze(0)
        latents_by_id[item["id"]] = lat  # shape (1, 16, H/8, W/8)

    optimizer = torch.optim.AdamW(
        [p for p in policy_pipe.transformer.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = (
        wandb is not None
        and not args.no_wandb
        and os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1")
    )
    if use_wandb:
        wandb.init(project="tfg-distillation", name="distill_run", config=vars(args))

    losses = []
    pbar = tqdm(range(args.max_steps), desc="distill")
    for step in pbar:
        batch = random.sample(items, args.batch_size)
        prompts_text = [b["prompt"] for b in batch]
        teacher_latents = torch.cat(
            [latents_by_id[b["id"]] for b in batch], dim=0
        ).to(device=device, dtype=torch.float32)

        pos_embeds, _neg, pos_pooled, _negp = _encode_prompts_cfg(
            policy_pipe, prompts_text, device
        )

        optimizer.zero_grad()
        loss = compute_fm_loss(
            policy_pipe.transformer,
            teacher_latents,
            pos_embeds,
            pos_pooled,
            policy_pipe.scheduler,
            torch.device(device),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy_pipe.transformer.parameters() if p.requires_grad],
            args.max_grad_norm,
        )
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % args.log_every == 0:
            recent = losses[-args.log_every:]
            pbar.set_postfix(loss=f"{sum(recent)/len(recent):.4f}")
            if use_wandb:
                wandb.log({"train/loss": loss.item(), "train/step": step + 1})

        if (step + 1) % args.save_every == 0:
            ckpt = output_dir / f"checkpoint-{step + 1}"
            policy_pipe.transformer.save_pretrained(str(ckpt))

    # Final checkpoint.
    final = output_dir / "final"
    policy_pipe.transformer.save_pretrained(str(final))
    print(f"saved distilled LoRA to {final}")

    # Persist a loss curve for result.md.
    with open(output_dir / "loss_curve.json", "w") as f:
        json.dump({"losses": losses}, f)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
