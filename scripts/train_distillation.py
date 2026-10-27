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
from scripts.train_state import (
    check_config,
    config_subset,
    load_train_state,
    resolve_resume,
    restore_rng,
    restore_trainable,
    save_lora_dir,
    save_train_state,
    truncate_jsonl,
)

# args that define the experiment; a resume with any of these changed is refused.
RESUME_CONFIG_KEYS = ("seed", "learning_rate", "batch_size", "max_grad_norm", "lora_rank", "lora_alpha")


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
    parser.add_argument("--wall_hours", type=float, default=None,
                        help="Stop after this many hours (matched regime: 4). Overrides --max_steps when hit first.")
    parser.add_argument("--save_every_minutes", type=float, default=0.0)
    parser.add_argument("--lora_fp32", action="store_true",
                        help="Keep LoRA params in fp32 (base bf16); recommended.")
    parser.add_argument("--resume", default="auto",
                        help="auto (default: continue from the latest resumable checkpoint in --output_dir), "
                             "none (fresh run), or a checkpoint-N directory.")
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

    if args.lora_fp32:
        for p_ in policy_pipe.transformer.parameters():
            if p_.requires_grad:
                p_.data = p_.data.float()

    use_wandb = (
        wandb is not None
        and not args.no_wandb
        and os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1")
    )
    if use_wandb:
        wandb.init(project="tfg-distillation", name="distill_run", config=vars(args))

    summary = train_distill(policy_pipe, items, latents_by_id, args, args.output_dir, device=device,
                            resume=args.resume, wandb_run=wandb if use_wandb else None)
    print(f"saved distilled LoRA to {summary['final_checkpoint']} ({summary['steps']} steps, "
          f"{summary['stop_reason']}, {summary['elapsed_s']:.0f}s)")

    if use_wandb:
        wandb.finish()


def train_distill(
    policy_pipe,
    items: list,
    latents_by_id: dict,
    args,
    output_dir: str | Path,
    device: str = "cuda",
    encode_fn=None,
    log_fn=print,
    resume: str | Path | None = "auto",
    wandb_run=None,
) -> dict:
    """Flow-matching LoRA loop over teacher latents. Stops at args.max_steps
    (<=0 = unbounded) or args.wall_hours, whichever first.

    encode_fn(list[str]) -> (pos, neg, pos_pooled, neg_pooled); defaults to the
    SD3 CFG encoder, injectable for the CPU test. resume as in train_draft.py:
    "auto" continues from the latest <output_dir>/checkpoint-N with a
    train_state.pt (exact replay: params, Adam moments, RNG, counters, budget).
    """
    if encode_fn is None:
        encode_fn = lambda texts: _encode_prompts_cfg(policy_pipe, texts, device)  # noqa: E731
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in policy_pipe.transformer.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters: LoRA not attached?")
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    losses: list[float] = []
    t0 = time.time()
    wall_budget = args.wall_hours * 3600.0 if args.wall_hours else None
    stop_reason = "max_steps"
    max_steps = args.max_steps if args.max_steps and args.max_steps > 0 else 10**9
    log_path = output_dir / "train_log.jsonl"
    step = 0            # optimizer steps completed
    resumed_from = None

    resume_dir = resolve_resume(resume, output_dir)
    if resume_dir is not None:
        state = load_train_state(resume_dir / "train_state.pt")
        check_config(state["config"], config_subset(args, RESUME_CONFIG_KEYS), RESUME_CONFIG_KEYS)
        n_restored = restore_trainable(policy_pipe.transformer, state["trainable"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])          # the sampler is the global `random` module here
        lp = state["loop"]
        step = int(lp["step"])
        losses = list(lp["losses"])
        t0 = time.time() - float(lp["elapsed_s"])
        kept = truncate_jsonl(log_path, step)
        resumed_from = str(resume_dir)
        log_fn(f"[distill] RESUMED from {resume_dir}: step {step}, {lp['elapsed_s']:.0f}s elapsed, "
               f"{n_restored} param tensors, log kept {kept} records")
    log_f = open(log_path, "a")
    last_time_ckpt = time.time()

    def _save_checkpoint(n: int) -> Path:
        ckpt_dir = output_dir / f"checkpoint-{n}"
        save_lora_dir(policy_pipe.transformer, ckpt_dir / "lora")
        save_train_state(
            ckpt_dir, policy_pipe.transformer, optimizer,
            loop={"step": n, "elapsed_s": round(time.time() - t0, 3), "losses": losses},
            config=config_subset(args, RESUME_CONFIG_KEYS),
        )
        return ckpt_dir / "lora"

    pbar = tqdm(total=max_steps if max_steps < 10**9 else None, initial=step, desc="distill",
                disable=log_fn is not print)
    while step < max_steps:
        if wall_budget is not None and (time.time() - t0) >= wall_budget:
            stop_reason = "wall_hours"
            break
        batch = random.sample(items, min(args.batch_size, len(items)))
        prompts_text = [b["prompt"] for b in batch]
        teacher_latents = torch.cat(
            [latents_by_id[b["id"]] for b in batch], dim=0
        ).to(device=device, dtype=torch.float32)

        pos_embeds, _neg, pos_pooled, _negp = encode_fn(prompts_text)

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
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        step += 1

        losses.append(loss.item())
        log_f.write(json.dumps({"step": step, "loss": losses[-1],
                                "elapsed_s": round(time.time() - t0, 1)}) + "\n")
        log_f.flush()
        pbar.update(1)
        if args.log_every and step % args.log_every == 0:
            recent = losses[-args.log_every:]
            pbar.set_postfix(loss=f"{sum(recent)/len(recent):.4f}")
            if wandb_run is not None:
                wandb_run.log({"train/loss": losses[-1], "train/step": step})

        now = time.time()
        by_step = args.save_every and step % args.save_every == 0
        by_time = args.save_every_minutes and (now - last_time_ckpt) >= args.save_every_minutes * 60
        if by_step or by_time:
            _save_checkpoint(step)
            last_time_ckpt = now
    pbar.close()
    log_f.close()

    # Final checkpoint (FlowGRPO layout <run>/checkpoint-N/lora, plus legacy `final`); weight-only.
    final = save_lora_dir(policy_pipe.transformer, output_dir / f"checkpoint-{step}" / "lora")
    save_lora_dir(policy_pipe.transformer, output_dir / "final")

    with open(output_dir / "loss_curve.json", "w") as f:
        json.dump({"losses": losses}, f)
    summary = {"steps": step, "stop_reason": stop_reason, "elapsed_s": round(time.time() - t0, 1),
               "n_teacher": len(items), "lr": args.learning_rate, "seed": args.seed,
               "loss_first50": sum(losses[:50]) / max(1, len(losses[:50])) if losses else None,
               "loss_last50": sum(losses[-50:]) / max(1, len(losses[-50:])) if losses else None,
               "final_checkpoint": str(final), "resumed_from": resumed_from}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    main()
