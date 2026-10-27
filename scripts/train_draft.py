#!/usr/bin/env python3
"""
DRaFT-K LoRA fine-tuning of SD3.5-M on a differentiable reward.

Why this arm exists
-------------------
The weights-or-sampling map has four cells: where the reward lives (weights vs
sampler) x what signal is used (scalar reward vs reward gradient).

    GRPO       = weights x scalar      (have)
    best-of-N  = sampler x scalar      (have)
    TFG        = sampler x gradient    (have)
    DRaFT      = weights x gradient    (this file)

Without the DRaFT cell the paper confounds location with signal: guidance
beating GRPO could be because the reward lives in the sampler, or just because
guidance gets the gradient. DRaFT trains the weights on the same gradient.

Algorithm (Clark et al. 2024, "Directly Fine-Tuning Diffusion Models on
Differentiable Rewards"): run the full denoising chain, backprop the reward
through only the last K Euler steps (K=1, DRaFT-1, by default), update LoRA.
No KL term by default: LoRA is the regulariser (as in DRaFT), and the study's
beta=0 result showed the upstream KL default was a handicap for GRPO. A
velocity-matching KL against a frozen reference stays behind --kl_coef for
ablation.

Differences from scripts/train_ddrl.py (the earlier null run on CLIP)
---------------------------------------------------------------------
  * reward selectable (clip | pickscore | aesthetic | ensemble | imagereward);
    the SAME predictor classes TFG guides on, so each reward has one source of
    truth across the training-free and training-based arms
  * lr default 1e-4 (LoRA scale). The null run used 1e-5, a plausible reason
    it did not move. Sweep it in the pilot before trusting any number.
  * kl_coef default 0. The null run used 0.05 velocity-KL, suspected to cancel
    the reward signal.
  * wall-clock budget (--wall_hours) so the arm is compute-matched to the
    FlowGRPO regime (4h on one H100); checkpoints by step and by time
  * LoRA params kept in fp32 (base bf16); gradient checkpointing and VAE
    slicing flags for 512^2 batch 4 on 80GB
  * seeded init noise and prompt order; JSON-lines log of reward per step for
    the budget-crossover figure
  * checkpoint layout <output_dir>/checkpoint-N/lora, matching FlowGRPO's, so
    eval_study_matrix.py --rl-lora and the pub discovery logic work unchanged

Reward sign: every predictor returns higher-is-better, so loss = -reward.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_ddrl import (  # noqa: E402
    _encode_prompts_cfg,
    _prompt_to_text,
    compute_velocity_kl,
    decode_and_score,
    rollout,
    setup_policy_model,
    setup_reference_model,
)
from scripts.train_state import (  # noqa: E402
    check_config,
    config_subset,
    load_train_state,
    resolve_resume,
    restore_rng,
    restore_trainable,
    save_train_state,
    truncate_jsonl,
)

REWARDS = ("clip", "pickscore", "aesthetic", "ensemble", "imagereward")

# cfg keys that define the experiment; a resume with any of these changed is refused.
RESUME_CONFIG_KEYS = (
    "seed", "lr", "weight_decay", "batch_size", "grad_accum", "grad_steps",
    "num_inference_steps", "guidance_scale", "height", "width", "kl_coef", "max_grad_norm",
)


# ─────────────────────────────────────────────────────────────────────────────
# Rewards + prompts
# ─────────────────────────────────────────────────────────────────────────────


def build_predictor(reward: str, device: str):
    """Same predictor classes eval_study_matrix.py uses for TFG guidance."""
    if reward == "clip":
        from scripts.clip_predictor import CLIPPromptPredictor
        return CLIPPromptPredictor(device=device)
    if reward == "pickscore":
        from scripts.pickscore_predictor import PickScorePromptPredictor
        return PickScorePromptPredictor(device=device)
    if reward == "aesthetic":
        from scripts.aesthetic_predictor import AestheticPredictor
        return AestheticPredictor(device=device)
    if reward == "ensemble":
        from scripts.ensemble_predictor import EnsemblePromptPredictor
        return EnsemblePromptPredictor(device=device)
    if reward == "imagereward":
        from scripts.imagereward_predictor import ImageRewardPredictor
        return ImageRewardPredictor(device=device)
    raise ValueError(f"unknown reward {reward!r}; choose from {REWARDS}")


def load_prompts(path: str | Path) -> list[str]:
    """JSON list of {prompt}/str, JSON {"prompts": [...]}, or one prompt per line."""
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            data = data.get("prompts", [])
        return [_prompt_to_text(d) for d in data]
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# One DRaFT step
# ─────────────────────────────────────────────────────────────────────────────


def draft_step(
    pipe,
    predictor,
    batch_text: list[str],
    embeds: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    latent_shape: tuple[int, int, int, int],
    cfg,
    generator: torch.Generator,
    device: str,
    ref_pipe=None,
):
    """Rollout with grad on the last cfg.grad_steps Euler steps, decode, score.

    Returns (loss, reward_mean_detached, kl_detached). The reward term is the
    DRaFT objective; the optional KL is the velocity-matching regulariser from
    train_ddrl.py, off unless cfg.kl_coef > 0 and a reference pipe is given.
    """
    pos, neg, pos_p, neg_p = embeds
    latents = rollout(
        pipe, pos, pos_p, neg, neg_p,
        latent_shape=latent_shape,
        num_inference_steps=cfg.num_inference_steps,
        grad_steps=cfg.grad_steps,
        guidance_scale=cfg.guidance_scale,
        generator=generator,
        device=device,
    )
    rewards = decode_and_score(pipe.vae, predictor, latents, batch_text)
    reward_mean = rewards.mean()
    loss = -reward_mean
    kl = torch.zeros((), device=rewards.device)
    if ref_pipe is not None and cfg.kl_coef > 0:
        kl = compute_velocity_kl(
            pipe.transformer, ref_pipe.transformer, latents.detach(),
            pos, pos_p, pipe.scheduler, device=device,
        )
        loss = loss + cfg.kl_coef * kl
    return loss, reward_mean.detach(), kl.detach()


def make_cached_encoder(pipe, device: str):
    """Encode each distinct prompt once and reuse (30 training prompts -> ~80MB).

    Removes the per-step text-encoder pass, which on 40GB boxes with T5 parked
    on CPU is a multi-second shuttle every step. Embeddings are prompt-only, so
    caching is exact.
    """
    cache: dict[str, tuple] = {}

    def encode(texts: list[str]):
        for t in texts:
            if t not in cache:
                pos, neg, pos_p, neg_p = _encode_prompts_cfg(pipe, [t], device)
                cache[t] = tuple(x.detach() for x in (pos, neg, pos_p, neg_p))
        parts = [cache[t] for t in texts]
        return tuple(torch.cat([pt[i] for pt in parts], dim=0) for i in range(4))

    return encode


def trainable_params(transformer) -> list[torch.nn.Parameter]:
    return [p for p in transformer.parameters() if p.requires_grad]


def save_lora(transformer, ckpt_dir: Path) -> Path:
    """<ckpt_dir>/lora, the FlowGRPO layout. Falls back to a state_dict for stubs."""
    d = ckpt_dir / "lora"
    d.mkdir(parents=True, exist_ok=True)
    if hasattr(transformer, "save_pretrained"):
        transformer.save_pretrained(str(d))
    else:
        torch.save(transformer.state_dict(), d / "state_dict.pt")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Train loop
# ─────────────────────────────────────────────────────────────────────────────


def train_draft(
    pipe,
    predictor,
    prompts: list[str],
    cfg,
    output_dir: str | Path,
    device: str = "cuda",
    encode_fn=None,
    ref_pipe=None,
    log_fn=print,
    resume: str | Path | None = "auto",
) -> dict:
    """DRaFT loop. Stops at cfg.max_steps or cfg.wall_hours, whichever first.

    resume: "auto" (default) continues from the latest <output_dir>/checkpoint-N
    that has a train_state.pt (fresh run if none); "none" forces a fresh run; a
    path resumes from that checkpoint. A resumed run replays exactly: trainable
    params, Adam moments, prompt sampler, RNG streams, counters and wall-clock
    budget are all restored (scripts/train_state.py).

    cfg fields: seed, lr, weight_decay, batch_size, grad_accum, grad_steps,
    num_inference_steps, guidance_scale, height, width, kl_coef, max_steps,
    wall_hours, max_grad_norm, save_every, save_every_minutes, log_every.
    encode_fn(list[str]) -> (pos, neg, pos_pooled, neg_pooled); defaults to the
    SD3 CFG encoder, injectable so the CPU smoke test can use stubs.
    """
    if not prompts:
        raise ValueError("no training prompts")
    if encode_fn is None:
        encode_fn = make_cached_encoder(pipe, device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    gen_device = "cuda" if str(device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=gen_device)

    params = trainable_params(pipe.transformer)
    if not params:
        raise RuntimeError("no trainable parameters: LoRA not attached?")
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    vae_scale = getattr(pipe, "vae_scale_factor", 8)
    latent_channels = pipe.transformer.config.in_channels
    latent_shape = (cfg.batch_size, latent_channels, cfg.height // vae_scale, cfg.width // vae_scale)

    log_path = output_dir / "train_log.jsonl"
    t0 = time.time()
    wall_budget = cfg.wall_hours * 3600.0 if cfg.wall_hours else None
    step, micro = 0, 0
    reward_hist: list[float] = []
    accum_rewards: list[float] = []
    stop_reason = "max_steps"
    # Collapse guard (reward hacking shows up as a cliff in the TRAINING reward:
    # the Oct-4 CLIP run fell 0.31 -> 0.19 within ~10 steps at step ~360 and
    # never produced a real image again). Track the smoothed reward; if it sits
    # more than `collapse_drop` below its running peak for `collapse_patience`
    # consecutive steps, stop and keep the peak checkpoint as the result.
    guard_on = getattr(cfg, "collapse_drop", 0.0) and getattr(cfg, "collapse_drop", 0.0) > 0
    guard_win = max(1, int(getattr(cfg, "collapse_window", 50)))
    guard_pat = max(1, int(getattr(cfg, "collapse_patience", 60)))
    peak_sm, peak_step, below = -1e9, 0, 0

    resumed_from = None
    resume_dir = resolve_resume(resume, output_dir)
    if resume_dir is not None:
        state = load_train_state(resume_dir / "train_state.pt")
        check_config(state["config"], config_subset(cfg, RESUME_CONFIG_KEYS), RESUME_CONFIG_KEYS)
        n_restored = restore_trainable(pipe.transformer, state["trainable"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], rng)
        lp = state["loop"]
        step, micro = int(lp["step"]), int(lp["micro"])
        reward_hist = list(lp["reward_hist"])
        peak_sm, peak_step, below = lp["peak_sm"], lp["peak_step"], lp["below"]
        t0 = time.time() - float(lp["elapsed_s"])   # wall budget counts all segments
        kept = truncate_jsonl(log_path, step)
        resumed_from = str(resume_dir)
        log_fn(f"[draft] RESUMED from {resume_dir}: step {step}, micro {micro}, "
               f"{lp['elapsed_s']:.0f}s elapsed, {n_restored} param tensors, log kept {kept} records")
    log_f = open(log_path, "a")
    last_time_ckpt = time.time()

    def _save_checkpoint(ckpt_dir: Path) -> None:
        save_lora(pipe.transformer, ckpt_dir)
        save_train_state(
            ckpt_dir, pipe.transformer, optimizer,
            loop={"step": step, "micro": micro, "elapsed_s": round(time.time() - t0, 3),
                  "reward_hist": reward_hist, "peak_sm": peak_sm, "peak_step": peak_step, "below": below},
            config=config_subset(cfg, RESUME_CONFIG_KEYS), py_rng=rng,
        )

    optimizer.zero_grad(set_to_none=True)
    while True:
        if cfg.max_steps is not None and step >= cfg.max_steps:
            stop_reason = "max_steps"
            break
        if wall_budget is not None and (time.time() - t0) >= wall_budget:
            stop_reason = "wall_hours"
            break

        # Sample prompts with replacement, as FlowGRPO's sampler effectively does
        # over a 30-prompt train set; seeded so a run is reproducible.
        batch_text = [rng.choice(prompts) for _ in range(cfg.batch_size)]
        embeds = encode_fn(batch_text)
        generator.manual_seed(cfg.seed * 1_000_003 + micro)

        loss, reward_mean, kl = draft_step(
            pipe, predictor, batch_text, embeds, latent_shape, cfg, generator, device, ref_pipe,
        )
        (loss / cfg.grad_accum).backward()
        micro += 1
        accum_rewards.append(reward_mean.item())

        if micro == 1:
            # Guard against the silent failure mode of reentrant gradient
            # checkpointing (no input requires grad -> param grads stay None):
            # the very first backward must reach the LoRA parameters.
            gsum = sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
            if gsum == 0.0:
                raise RuntimeError(
                    "first backward produced no gradient on trainable params; "
                    "check gradient checkpointing / LoRA attachment before spending GPU time")

        if micro % cfg.grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            r = sum(accum_rewards) / len(accum_rewards)
            accum_rewards = []
            reward_hist.append(r)
            rec = {
                "step": step, "micro": micro, "reward": r, "loss": float(loss.item()),
                "kl": float(kl.item()), "grad_norm": float(grad_norm),
                "elapsed_s": round(time.time() - t0, 1),
            }
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if cfg.log_every and step % cfg.log_every == 0:
                log_fn(f"[draft] step {step}  reward {r:.4f}  gn {float(grad_norm):.3f}  "
                       f"{rec['elapsed_s']}s")

            if guard_on and len(reward_hist) >= guard_win:
                sm = sum(reward_hist[-guard_win:]) / guard_win
                if sm > peak_sm:
                    peak_sm, peak_step, below = sm, step, 0
                elif sm < peak_sm - cfg.collapse_drop:
                    below += 1
                    if below >= guard_pat:
                        log_fn(f"[draft] COLLAPSE GUARD: smoothed reward {sm:.4f} has been > {cfg.collapse_drop} "
                               f"below its peak {peak_sm:.4f} (step {peak_step}) for {below} steps; stopping")
                        stop_reason = "collapse_guard"
                        break
                else:
                    below = 0

            now = time.time()
            by_step = bool(cfg.save_every) and step % cfg.save_every == 0
            by_time = bool(cfg.save_every_minutes) and (now - last_time_ckpt) >= cfg.save_every_minutes * 60
            if by_step or by_time:
                _save_checkpoint(output_dir / f"checkpoint-{step}")
                last_time_ckpt = now

    log_f.close()
    final_dir = None
    if step > 0:
        final_dir = save_lora(pipe.transformer, output_dir / f"checkpoint-{step}")

    n = max(1, min(10, len(reward_hist)))
    summary = {
        "steps": step, "micro_batches": micro, "stop_reason": stop_reason,
        "peak_smoothed_reward": None if peak_sm == -1e9 else peak_sm, "peak_step": peak_step,
        "elapsed_s": round(time.time() - t0, 1),
        "reward_first10": (sum(reward_hist[:n]) / n) if reward_hist else None,
        "reward_last10": (sum(reward_hist[-n:]) / n) if reward_hist else None,
        "final_checkpoint": str(final_dir) if final_dir else None,
        "resumed_from": resumed_from,
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────


def _enable_gradient_checkpointing(transformer) -> bool:
    """PEFT wraps the model: PeftModel.base_model.model is the diffusers transformer."""
    m = transformer
    for attr in ("base_model", "model"):
        m = getattr(m, attr, m)
    fn = getattr(m, "enable_gradient_checkpointing", None)
    if fn is None:
        return False
    fn()
    return True


def main():
    ap = argparse.ArgumentParser(description="DRaFT-K LoRA fine-tune of SD3.5-M on a differentiable reward")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--reward", choices=REWARDS, required=True)
    ap.add_argument("--prompts_file", default="scripts/cloud/prompts_smoke.json",
                    help="JSON list or one-per-line txt. Default = the 30 FlowGRPO training prompts.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    # budget
    ap.add_argument("--wall_hours", type=float, default=None, help="Stop after this many hours (matched regime: 4).")
    ap.add_argument("--max_steps", type=int, default=None, help="Stop after this many optimizer steps.")
    # DRaFT
    ap.add_argument("--grad_steps", type=int, default=1, help="K in DRaFT-K: Euler steps with grad.")
    ap.add_argument("--num_inference_steps", type=int, default=20)
    ap.add_argument("--guidance_scale", type=float, default=4.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    # optimisation
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--kl_coef", type=float, default=0.0, help="Velocity-KL vs frozen reference; 0 = off (default).")
    ap.add_argument("--collapse_drop", type=float, default=0.04,
                    help="Stop when the smoothed train reward sits this far below its peak (0 = off).")
    ap.add_argument("--collapse_window", type=int, default=50)
    ap.add_argument("--collapse_patience", type=int, default=60)
    # memory
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--vae_slicing", action="store_true", help="Decode one sample at a time (grads still flow).")
    ap.add_argument("--t5_cpu", action="store_true", help="Park T5-XXL on CPU (24-40GB boxes).")
    ap.add_argument("--no_lora_fp32", action="store_true", help="Keep LoRA params in bf16 (default upcasts to fp32).")
    # logging / ckpt
    ap.add_argument("--save_every", type=int, default=0, help="Checkpoint every N steps (0 = off).")
    ap.add_argument("--save_every_minutes", type=float, default=30.0, help="Checkpoint every M minutes (0 = off).")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--resume", default="auto",
                    help="auto (default: continue from the latest resumable checkpoint in --output_dir), "
                         "none (fresh run), or a checkpoint-N directory.")
    args = ap.parse_args()

    if args.wall_hours is None and args.max_steps is None:
        ap.error("give --wall_hours and/or --max_steps")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = load_prompts(args.prompts_file)
    print(f"[draft] reward={args.reward}  prompts={len(prompts)} from {args.prompts_file}  seed={args.seed}")

    pipe = setup_policy_model(
        args.model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        device=device, t5_cpu=args.t5_cpu,
    )
    pipe.transformer.train()
    if not args.no_lora_fp32:
        for p in trainable_params(pipe.transformer):
            p.data = p.data.float()
    if args.gradient_checkpointing:
        ok = _enable_gradient_checkpointing(pipe.transformer)
        print(f"[draft] gradient checkpointing: {'on' if ok else 'unavailable'}")
    if args.vae_slicing and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    ref_pipe = None
    if args.kl_coef > 0:
        ref_pipe = setup_reference_model(args.model, device=device, t5_cpu=args.t5_cpu)

    predictor = build_predictor(args.reward, device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    summary = train_draft(pipe, predictor, prompts, args, out, device=device, ref_pipe=ref_pipe,
                          resume=args.resume)
    print(f"[draft] done: {summary['steps']} steps in {summary['elapsed_s']}s ({summary['stop_reason']}); "
          f"reward first10 {summary['reward_first10']} -> last10 {summary['reward_last10']}; "
          f"final {summary['final_checkpoint']}")


if __name__ == "__main__":
    main()
