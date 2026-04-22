#!/usr/bin/env python3
"""
LoRA fine-tuning for SD3.5-M with a differentiable reward + velocity-matching KL.

Loss = -reward_weight * mean(R_CLIP(x_0, prompt))
     + kl_coef      * mean(||v_theta(x_t, t) - v_ref(x_t, t)||^2)

Honesty about the algorithm
---------------------------
The DDRL paper's code is not public. The previous version of this file called
itself DDRL but its "log prob" was an ad-hoc velocity-residual MSE on a single
noise draw — not a valid log-probability estimator, not a valid KL estimate.
That loss would not train the policy toward higher reward in any reliable way.

This rewrite is a principled first pass that:

  (1) Reward term uses DRaFT-lite: gradient is enabled only on the last
      `grad_steps` Euler steps of the denoising trajectory. Earlier steps run
      under torch.no_grad() so memory stays bounded and we do not need to
      backprop through the full 20-40-step ODE. With `grad_steps=1` this is
      effectively "reward-on-final-step" — the cheapest form of direct reward
      fine-tuning. DRaFT (Clark et al., 2024) validates this works for image
      diffusion LoRA fine-tunes.

  (2) KL term is a velocity-matching consistency loss between policy and a
      frozen reference at a random sigma. This regularizes the policy toward
      the reference's velocity field — acts as a KL penalty in practice,
      though it is not the exact forward-KL from the DDRL paper docstring.
      If the DDRL paper is published, the term can be swapped without
      touching orchestration.

  (3) Reward is CLIP-similarity to the prompt. GenEval's mmdetection pipeline
      is not differentiable; the same substitution is used on the TFG side
      (see scripts/clip_predictor.py). GenEval remains the downstream
      evaluation metric — training just cannot guide on it directly.

Renaming later is easy; producing garbage checkpoints under a prestigious
name is not. The CLI surface is unchanged so the sbatch scripts still work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# wandb is optional — missing module treated as WANDB_DISABLED. Keeps the CPU
# smoke test and any no-network environment runnable without installing wandb.
try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.clip_predictor import CLIPPromptPredictor


# ─────────────────────────────────────────────────────────────────────────────
# Config + prompts
# ─────────────────────────────────────────────────────────────────────────────


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_training_prompts(prompts_file: str, max_prompts: int | None = None) -> list:
    if prompts_file.endswith(".json"):
        with open(prompts_file, "r") as f:
            data = json.load(f)
            prompts = data if isinstance(data, list) else data.get("prompts", [])
    elif prompts_file.endswith(".jsonl"):
        prompts = []
        with open(prompts_file, "r") as f:
            for line in f:
                prompts.append(json.loads(line))
    else:
        raise ValueError(f"Unknown prompt format: {prompts_file}")
    if max_prompts:
        prompts = prompts[:max_prompts]
    return prompts


def _prompt_to_text(p) -> str:
    if isinstance(p, dict):
        return p.get("prompt", p.get("text", str(p)))
    return str(p)


# ─────────────────────────────────────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────────────────────────────────────


def setup_policy_model(
    model_name: str,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    device: str = "cuda",
    t5_cpu: bool = False,
):
    """Load SD3.5-M and attach LoRA to the transformer (only).

    Text encoders and VAE stay frozen and unmodified. LoRA targets the
    attention projections of the transformer — standard for SD3 LoRA fine-tunes.

    t5_cpu: park T5-XXL on CPU (saves ~9.5 GB). Required on 24 GB A10. The
    training loop shuttles T5 to GPU for encode_prompt and back after.
    """
    from diffusers import StableDiffusion3Pipeline
    from peft import LoraConfig, get_peft_model

    print(f"[policy] loading {model_name}")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)

    if t5_cpu and getattr(pipe, "text_encoder_3", None) is not None:
        pipe.text_encoder_3.to("cpu")
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"  [mem] T5 offloaded: {free/1e9:.1f}GB free of {total/1e9:.1f}GB")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.print_trainable_parameters()

    # Text encoders and VAE are frozen for RL fine-tuning.
    for module in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        if module is None:
            continue
        module.requires_grad_(False)
        module.eval()

    return pipe


def setup_reference_model(model_name: str, device: str = "cuda", t5_cpu: bool = False):
    """Frozen reference (base SD3.5-M, no LoRA).

    t5_cpu applies the same T5-XXL offload as setup_policy_model. The reference
    model's T5 isn't used during the velocity-KL term (which only needs the
    transformer at a random sigma), so it can live on CPU permanently — it's
    only touched during rollouts via encode_prompt, and we already use the
    policy's encoder cache by passing pre-computed embeds into compute_velocity_kl.
    """
    from diffusers import StableDiffusion3Pipeline

    print(f"[ref] loading {model_name} (frozen)")
    ref = StableDiffusion3Pipeline.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    ref.transformer.requires_grad_(False)
    ref.transformer.eval()
    for module in (ref.vae, ref.text_encoder, ref.text_encoder_2, ref.text_encoder_3):
        if module is None:
            continue
        module.requires_grad_(False)
        module.eval()
    if t5_cpu and getattr(ref, "text_encoder_3", None) is not None:
        ref.text_encoder_3.to("cpu")
        torch.cuda.empty_cache()
    return ref


def setup_reward_predictor(device: str = "cuda") -> CLIPPromptPredictor:
    """CLIP-similarity reward. Differentiable wrt image.

    Same predictor class used on the TFG side — keep one source of truth so
    any bug in the reward shape/norm shows up in both places at once.
    """
    return CLIPPromptPredictor(device=device, torch_dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout + reward + KL
# ─────────────────────────────────────────────────────────────────────────────


def _encode_prompts_cfg(pipe, prompts: list[str], device: str):
    """Encode prompts for CFG denoising. Returns (pos, neg) embed pairs.

    Uses pipe.encode_prompt (SD3 signature: 4 outputs). CFG needs both pos
    and neg; we keep them separate so the caller can choose CFG on/off.

    If T5 is parked on CPU (24GB memory budget), we shuttle it to GPU for the
    encode pass and back afterwards — each prompt batch costs ~2s of wall for
    the moves. Undefined or missing T5 is handled gracefully.
    """
    t5 = getattr(pipe, "text_encoder_3", None)
    t5_on_cpu = t5 is not None and next(t5.parameters()).device.type == "cpu"
    if t5_on_cpu:
        t5.to(device)
    try:
        with torch.no_grad():
            pos_embeds, neg_embeds, pos_pooled, neg_pooled = pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_3=None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
    finally:
        if t5_on_cpu and t5 is not None:
            t5.to("cpu")
            torch.cuda.empty_cache()
    return pos_embeds, neg_embeds, pos_pooled, neg_pooled


def _transformer_forward(
    transformer, latents, t_batch, encoder_hidden_states, pooled_projections
) -> torch.Tensor:
    """Unified transformer call that returns velocity in fp32.

    Casts hidden_states + embeds to the transformer's dtype (usually bf16) then
    upcasts the output to fp32 for numerically stable loss math.
    """
    v = transformer(
        hidden_states=latents.to(transformer.dtype),
        timestep=t_batch,
        encoder_hidden_states=encoder_hidden_states.to(transformer.dtype),
        pooled_projections=pooled_projections.to(transformer.dtype),
        return_dict=False,
    )[0].to(torch.float32)
    return v


def rollout(
    pipe,
    pos_embeds: torch.Tensor,
    pos_pooled: torch.Tensor,
    neg_embeds: torch.Tensor,
    neg_pooled: torch.Tensor,
    latent_shape: tuple[int, int, int, int],
    num_inference_steps: int,
    grad_steps: int,
    guidance_scale: float,
    generator: torch.Generator,
    device: str,
) -> torch.Tensor:
    """Custom Euler rollout with grad scope limited to the last `grad_steps`.

    Early steps run under torch.no_grad() so memory is bounded regardless of
    num_inference_steps. Only the trailing `grad_steps` contribute to the
    reward gradient — this is DRaFT-lite, not full BPTT-through-ODE.

    Returns final latents of shape latent_shape (typically (B, 16, H/8, W/8)
    for SD3). Caller decodes via VAE for reward computation.
    """
    scheduler = pipe.scheduler
    transformer = pipe.transformer

    # Initialize scheduler state for this rollout.
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)

    latents = torch.randn(
        latent_shape, generator=generator, device=device, dtype=torch.float32
    ) * sigmas[0]  # scale to initial sigma — matches FlowMatchEuler init

    n = len(timesteps)
    first_grad_idx = max(0, n - grad_steps)

    for i, t in enumerate(timesteps):
        t_batch = t.expand(latents.shape[0]).to(device)

        def _velocity():
            if guidance_scale > 1.0:
                # CFG: duplicate batch for pos/neg, combine.
                latent_in = torch.cat([latents, latents], dim=0)
                embeds_in = torch.cat([neg_embeds, pos_embeds], dim=0)
                pooled_in = torch.cat([neg_pooled, pos_pooled], dim=0)
                t_in = t_batch.repeat(2)
                v = _transformer_forward(transformer, latent_in, t_in, embeds_in, pooled_in)
                v_neg, v_pos = v.chunk(2, dim=0)
                return v_neg + guidance_scale * (v_pos - v_neg)
            else:
                return _transformer_forward(transformer, latents, t_batch, pos_embeds, pos_pooled)

        if i < first_grad_idx:
            with torch.no_grad():
                v = _velocity()
        else:
            v = _velocity()

        # Euler step: x_{t-1} = x_t + dt * v, dt = sigma_next - sigma
        sigma_now = sigmas[i]
        sigma_next = sigmas[i + 1] if (i + 1) < len(sigmas) else torch.zeros_like(sigma_now)
        dt = (sigma_next - sigma_now).to(latents.dtype)
        latents = latents + dt * v

    return latents


def decode_and_score(
    vae,
    predictor: CLIPPromptPredictor,
    latents: torch.Tensor,
    prompts_text: list[str],
) -> torch.Tensor:
    """Decode latents through VAE and score with CLIP. Differentiable."""
    # SD3 VAE: encode returns (latent - shift) * scale, so decode reverses.
    scaled = latents / vae.config.scaling_factor + vae.config.shift_factor
    image = vae.decode(scaled.to(vae.dtype), return_dict=False)[0].to(torch.float32)
    return predictor(image, prompts_text)  # (B,)


def compute_velocity_kl(
    policy_transformer,
    ref_transformer,
    x0_detached: torch.Tensor,
    pos_embeds: torch.Tensor,
    pos_pooled: torch.Tensor,
    scheduler,
    device: str,
) -> torch.Tensor:
    """Velocity-matching KL term.

    Sample a random sigma in the interior (avoid extremes), diffuse x_0 to x_t,
    compute v_policy and v_ref at that x_t, return mean squared difference.
    Policy is the only graph that retains grad; ref output is detached.
    """
    B = x0_detached.shape[0]
    # Sample sigma ~ Uniform(0.1, 0.9) — avoids near-clean (0) and near-noise (1)
    # where velocity fields are either trivial or dominated by noise.
    sigma = torch.rand(B, device=device) * 0.8 + 0.1
    noise = torch.randn_like(x0_detached)
    sigma_expanded = sigma.view(-1, 1, 1, 1)
    x_t = sigma_expanded * noise + (1.0 - sigma_expanded) * x0_detached

    # Timestep in the scheduler's integer-ish space: sigma * num_train.
    num_train = int(scheduler.config.num_train_timesteps)
    t_batch = (sigma * num_train).to(device)

    with torch.no_grad():
        v_ref = _transformer_forward(
            ref_transformer, x_t, t_batch, pos_embeds, pos_pooled
        )

    v_policy = _transformer_forward(
        policy_transformer, x_t, t_batch, pos_embeds, pos_pooled
    )
    return F.mse_loss(v_policy, v_ref)


# ─────────────────────────────────────────────────────────────────────────────
# Train loop
# ─────────────────────────────────────────────────────────────────────────────


def train_ddrl(
    policy_pipe,
    ref_pipe,
    predictor: CLIPPromptPredictor,
    prompts: list,
    config: dict,
    output_dir: str,
    max_steps: int | None = None,
    device: str = "cuda",
):
    ddrl_cfg = config.get("ddrl", {})
    gen_cfg = config.get("generation", {})

    lr = ddrl_cfg.get("learning_rate", 1e-5)
    kl_coef = ddrl_cfg.get("kl_coef", 0.05)
    reward_weight = ddrl_cfg.get("reward_weight", 1.0)
    rollout_batch_size = ddrl_cfg.get("rollout_batch_size", 4)
    save_steps = ddrl_cfg.get("save_steps", 100)
    num_epochs = ddrl_cfg.get("num_train_epochs", 1)
    grad_accum = ddrl_cfg.get("gradient_accumulation_steps", 1)
    max_grad_norm = ddrl_cfg.get("max_grad_norm", 1.0)
    grad_steps = ddrl_cfg.get("grad_steps", 1)  # DRaFT-lite depth
    use_ref = ddrl_cfg.get("use_reference_kl", True)

    num_inference_steps = gen_cfg.get("num_inference_steps", 20)
    guidance_scale = gen_cfg.get("guidance_scale", 4.5)
    height = gen_cfg.get("height", 512)
    width = gen_cfg.get("width", 512)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SD3 latent channels = 16; spatial = input / vae_scale_factor (8).
    vae_scale = policy_pipe.vae_scale_factor if hasattr(policy_pipe, "vae_scale_factor") else 8
    latent_channels = policy_pipe.transformer.config.in_channels
    latent_shape = (
        rollout_batch_size,
        latent_channels,
        height // vae_scale,
        width // vae_scale,
    )

    optimizer = torch.optim.AdamW(
        [p for p in policy_pipe.transformer.parameters() if p.requires_grad],
        lr=lr,
    )

    generator = torch.Generator(device=device)
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        import random

        epoch_prompts = list(prompts)
        random.shuffle(epoch_prompts)
        num_batches = len(epoch_prompts) // rollout_batch_size

        pbar = tqdm(range(num_batches), desc=f"epoch {epoch+1}/{num_epochs}")
        for batch_idx in pbar:
            if max_steps is not None and global_step >= max_steps:
                break

            generator.manual_seed(42 + 1000 * epoch + batch_idx)

            batch = epoch_prompts[
                batch_idx * rollout_batch_size : (batch_idx + 1) * rollout_batch_size
            ]
            batch_text = [_prompt_to_text(p) for p in batch]

            pos_embeds, neg_embeds, pos_pooled, neg_pooled = _encode_prompts_cfg(
                policy_pipe, batch_text, device
            )

            # ── Rollout (grad only on last `grad_steps`) ───────────────────────
            latents = rollout(
                policy_pipe,
                pos_embeds, pos_pooled, neg_embeds, neg_pooled,
                latent_shape=latent_shape,
                num_inference_steps=num_inference_steps,
                grad_steps=grad_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                device=device,
            )

            # ── Reward (differentiable through last `grad_steps` of ODE) ──────
            rewards = decode_and_score(policy_pipe.vae, predictor, latents, batch_text)
            reward_mean = rewards.mean()
            reward_loss = -reward_weight * reward_mean

            # ── Velocity-matching KL regularizer ──────────────────────────────
            if use_ref and ref_pipe is not None and kl_coef > 0:
                kl_term = compute_velocity_kl(
                    policy_pipe.transformer,
                    ref_pipe.transformer,
                    latents.detach(),
                    pos_embeds,
                    pos_pooled,
                    policy_pipe.scheduler,
                    device=device,
                )
            else:
                kl_term = torch.zeros((), device=device)

            loss = reward_loss + kl_coef * kl_term
            (loss / grad_accum).backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy_pipe.transformer.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                log = {
                    "train/reward": reward_mean.item(),
                    "train/kl_velocity": kl_term.item(),
                    "train/loss": loss.item(),
                    "train/step": global_step,
                    "train/epoch": epoch,
                }
                if wandb is not None and wandb.run:
                    wandb.log(log)
                pbar.set_postfix(r=f"{reward_mean.item():.3f}", kl=f"{kl_term.item():.3f}")

                if global_step % save_steps == 0:
                    ckpt = output_dir / f"checkpoint-{global_step}"
                    policy_pipe.transformer.save_pretrained(str(ckpt))

        if max_steps is not None and global_step >= max_steps:
            break

    final = output_dir / "final"
    policy_pipe.transformer.save_pretrained(str(final))
    print(f"saved LoRA to {final}")
    return global_step


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="SD3.5-M LoRA fine-tune: CLIP reward + velocity-matching KL"
    )
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/ddrl")
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Hard cap on optimizer steps (overrides num_train_epochs).")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_ref_model", action="store_true",
                        help="Skip loading the reference model (no KL term). "
                             "Useful for single-GPU memory-constrained runs.")
    parser.add_argument("--t5_cpu", action="store_true",
                        help="Park T5-XXL on CPU, shuttle to GPU per encode_prompt. "
                             "Saves ~9.5 GB. Required on 24 GB A10.")
    args = parser.parse_args()

    config = load_config(args.config)
    ddrl_cfg = config.get("ddrl", {})
    if args.no_ref_model:
        ddrl_cfg = dict(ddrl_cfg)
        ddrl_cfg["use_reference_kl"] = False
        config = dict(config)
        config["ddrl"] = ddrl_cfg

    use_wandb = (
        wandb is not None
        and not args.no_wandb
        and os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1")
    )
    if use_wandb:
        wandb.init(
            project=config.get("logging", {}).get("wandb", {}).get("project", "tfg-rl-baselines"),
            name="ddrl_training",
            config={"ddrl": ddrl_cfg, **{k: v for k, v in config.items() if k != "ddrl"}},
        )

    prompts_file = args.prompts_file or config.get("geneval", {}).get("prompts_file")
    if prompts_file and Path(prompts_file).exists():
        prompts = load_training_prompts(prompts_file, args.max_prompts)
    else:
        print("warning: prompts file not found, using fallback")
        prompts = [
            {"id": i, "prompt": p} for i, p in enumerate([
                "A red apple on a wooden table",
                "Two cats sitting on a sofa",
                "A blue car parked next to a green tree",
                "Three birds flying over a lake",
                "A yellow banana next to a green bottle",
            ])
        ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config.get("model", {}).get("name", "stabilityai/stable-diffusion-3.5-medium")

    policy_pipe = setup_policy_model(
        model_name,
        lora_rank=ddrl_cfg.get("lora_rank", 64),
        lora_alpha=ddrl_cfg.get("lora_alpha", 128),
        device=device,
        t5_cpu=args.t5_cpu,
    )
    ref_pipe = None if args.no_ref_model else setup_reference_model(model_name, device=device, t5_cpu=args.t5_cpu)
    predictor = setup_reward_predictor(device=device)

    total_steps = train_ddrl(
        policy_pipe, ref_pipe, predictor, prompts,
        config, args.output_dir,
        max_steps=args.max_steps, device=device,
    )
    print(f"done. total optimizer steps: {total_steps}")
    if wandb is not None and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
