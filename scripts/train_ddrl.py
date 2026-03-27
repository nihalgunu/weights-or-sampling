#!/usr/bin/env python3
"""
DDRL (Direct Distillation from Reward Learning) training for SD3.5-M.

Algorithm: Forward KL divergence + reward maximization
  Loss = -reward_weight * E[R(x)] + kl_coef * KL_forward(p_theta || p_ref)
  where KL_forward = E_{x~p_ref}[log p_ref(x) - log p_theta(x)]

Key difference from FlowGRPO:
- No group relative comparison (no GRPO ranking)
- Uses forward KL (mode-covering) instead of reverse KL (mode-seeking)
- Reference model stays frozen; policy model updates toward high-reward regions
  while staying close to the reference in the forward KL sense
- Empirically shows less reward hacking than GRPO-style methods

Reference: "DDRL" paper (adapt from paper; repo not yet publicly released)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
import yaml
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "repos/geneval"))


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_training_prompts(prompts_file: str, max_prompts: int = None) -> list:
    """Load prompts for RL training (GenEval prompt set)."""
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


def setup_policy_model(model_name: str, lora_rank: int = 64, lora_alpha: int = 128, device: str = "cuda"):
    """Load SD3.5-M and attach LoRA for policy updates."""
    from diffusers import StableDiffusion3Pipeline
    from peft import LoraConfig, get_peft_model

    print(f"Loading policy model: {model_name}")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)

    # Apply LoRA only to the transformer (denoiser)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.print_trainable_parameters()

    return pipe


def setup_reference_model(model_name: str, device: str = "cuda"):
    """Load frozen reference model (base SD3.5-M, no LoRA)."""
    from diffusers import StableDiffusion3Pipeline

    print("Loading frozen reference model...")
    ref_pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    ref_pipe = ref_pipe.to(device)

    # Freeze all reference model parameters
    for param in ref_pipe.transformer.parameters():
        param.requires_grad = False

    return ref_pipe


def setup_geneval_reward(detector_path: str):
    """
    Set up GenEval detector as a reward function.

    Returns a callable that scores a batch of PIL images against their prompts.
    Uses the GenEval mmdetection-based detector to count objects and check
    spatial relations / attribute binding.
    """
    geneval_dir = PROJECT_ROOT / "repos/geneval"

    if not geneval_dir.exists():
        print("Warning: GenEval repo not found. Using dummy reward (always 0.5).")
        return lambda images, prompts: [0.5] * len(images)

    try:
        sys.path.insert(0, str(geneval_dir))
        from evaluation.evaluate_images import build_detector, evaluate_single

        detector = build_detector(detector_path)
        print("GenEval detector loaded for reward.")

        def reward_fn(images: list, prompts: list) -> list:
            scores = []
            for img, prompt in zip(images, prompts):
                try:
                    result = evaluate_single(detector, img, prompt)
                    scores.append(float(result.get("score", 0.0)))
                except Exception as e:
                    print(f"Warning: reward eval failed: {e}")
                    scores.append(0.0)
            return scores

        return reward_fn

    except ImportError as e:
        print(f"Warning: Could not import GenEval reward: {e}. Using dummy reward.")
        return lambda images, prompts: [0.5] * len(images)


def compute_log_prob_flow(pipe, latents, prompts, num_inference_steps: int = 20, device: str = "cuda") -> torch.Tensor:
    """
    Estimate log p_theta(x) for flow matching (SD3.5-M) by computing
    the negative reconstruction loss along a short ODE trajectory.

    For flow matching: v_theta(x_t, t) is the velocity field.
    Log-likelihood is approximated via the change-of-variables formula:
      log p(x_0) ≈ log p(x_T) - integral of div(v_theta) dt
    We approximate this with the L2 norm of the velocity residual on
    a subset of timesteps (sufficient for relative KL computation).

    Returns shape: (batch_size,)
    """
    pipe.transformer.eval()

    batch_size = latents.shape[0]
    log_probs = torch.zeros(batch_size, device=device, dtype=torch.float32)

    # Encode prompts
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_3=None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        )

    # Sample a subset of timesteps and compute velocity residuals
    timesteps_to_eval = torch.linspace(0.1, 0.9, steps=5, device=device)

    for t in timesteps_to_eval:
        t_batch = t.expand(batch_size)
        # Add noise at timestep t
        noise = torch.randn_like(latents)
        noisy_latents = (1 - t) * latents + t * noise

        with torch.no_grad():
            target_velocity = noise - latents  # Flow matching target

        with torch.enable_grad():
            pred_velocity = pipe.transformer(
                hidden_states=noisy_latents.to(pipe.transformer.dtype),
                timestep=t_batch,
                encoder_hidden_states=prompt_embeds.to(pipe.transformer.dtype),
                pooled_projections=pooled_prompt_embeds.to(pipe.transformer.dtype),
                return_dict=False,
            )[0].float()

        # Velocity residual as a proxy for log prob contribution
        residual = F.mse_loss(pred_velocity, target_velocity.float(), reduction="none")
        log_probs -= residual.mean(dim=[1, 2, 3])  # (batch,)

    return log_probs / len(timesteps_to_eval)


def generate_batch(pipe, prompts: list, num_inference_steps: int = 20, seed: int = 42, device: str = "cuda"):
    """Generate a batch of images and return images + latents."""
    generator = torch.Generator(device=device).manual_seed(seed)

    output = pipe(
        prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=4.5,
        generator=generator,
        output_type="pil",
    )
    images = output.images

    # Also get latents via a second encode pass for log-prob computation
    with torch.no_grad():
        latents = pipe.vae.encode(
            torch.stack([
                pipe.image_processor.preprocess(img).squeeze(0)
                for img in images
            ]).to(device, dtype=pipe.vae.dtype)
        ).latent_dist.sample()
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    return images, latents.detach()


def train_ddrl(
    policy_pipe,
    ref_pipe,
    reward_fn,
    prompts: list,
    config: dict,
    output_dir: str,
    device: str = "cuda",
):
    """
    Main DDRL training loop.

    Loss = -reward_weight * mean(rewards) + kl_coef * KL_forward(p_theta || p_ref)
    KL_forward ≈ mean(log_prob_ref - log_prob_theta)  [estimated from velocity residuals]
    """
    ddrl_cfg = config.get("ddrl", {})
    lr = ddrl_cfg.get("learning_rate", 1e-5)
    kl_coef = ddrl_cfg.get("kl_coef", 0.05)
    reward_weight = ddrl_cfg.get("reward_weight", 1.0)
    rollout_batch_size = ddrl_cfg.get("rollout_batch_size", 8)
    save_steps = ddrl_cfg.get("save_steps", 500)
    num_epochs = ddrl_cfg.get("num_train_epochs", 1)
    grad_accum = ddrl_cfg.get("gradient_accumulation_steps", 4)
    max_grad_norm = ddrl_cfg.get("max_grad_norm", 1.0)
    num_inference_steps = config.get("generation", {}).get("num_inference_steps", 20)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only optimize LoRA parameters
    optimizer = torch.optim.AdamW(
        [p for p in policy_pipe.transformer.parameters() if p.requires_grad],
        lr=lr,
    )

    global_step = 0
    policy_pipe.transformer.train()

    for epoch in range(num_epochs):
        # Shuffle prompts each epoch
        import random
        epoch_prompts = prompts.copy()
        random.shuffle(epoch_prompts)

        num_batches = len(epoch_prompts) // rollout_batch_size
        optimizer.zero_grad()

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_prompts = epoch_prompts[batch_idx * rollout_batch_size:(batch_idx + 1) * rollout_batch_size]
            batch_text = [
                p.get("prompt", p.get("text", str(p))) if isinstance(p, dict) else str(p)
                for p in batch_prompts
            ]

            seed = 42 + global_step

            # ── Rollout: generate images from policy ──────────────────────────
            policy_pipe.transformer.eval()
            with torch.no_grad():
                images, latents = generate_batch(
                    policy_pipe, batch_text,
                    num_inference_steps=num_inference_steps,
                    seed=seed, device=device,
                )

            # ── Reward signal ────────────────────────────────────────────────
            rewards = reward_fn(images, batch_text)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            mean_reward = rewards_t.mean()

            # ── Forward KL: log p_ref - log p_theta ──────────────────────────
            # Use reference model latents as the "reference samples"
            with torch.no_grad():
                ref_images, ref_latents = generate_batch(
                    ref_pipe, batch_text,
                    num_inference_steps=num_inference_steps,
                    seed=seed, device=device,
                )
                log_prob_ref = compute_log_prob_flow(
                    ref_pipe, ref_latents, batch_text,
                    num_inference_steps=5, device=device,
                )

            policy_pipe.transformer.train()
            log_prob_theta = compute_log_prob_flow(
                policy_pipe, ref_latents, batch_text,
                num_inference_steps=5, device=device,
            )

            # KL_forward = E_{x~p_ref}[log p_ref - log p_theta]
            kl_forward = (log_prob_ref - log_prob_theta).mean()

            # ── Total loss ───────────────────────────────────────────────────
            loss = -reward_weight * mean_reward + kl_coef * kl_forward
            loss = loss / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy_pipe.transformer.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # ── Logging ──────────────────────────────────────────────────
                if wandb.run:
                    wandb.log({
                        "train/reward": mean_reward.item(),
                        "train/kl_forward": kl_forward.item(),
                        "train/loss": loss.item() * grad_accum,
                        "train/step": global_step,
                        "train/epoch": epoch,
                    })

                print(
                    f"  Step {global_step} | reward={mean_reward.item():.4f} "
                    f"| kl={kl_forward.item():.4f} | loss={loss.item() * grad_accum:.4f}"
                )

                # ── Checkpoint ───────────────────────────────────────────────
                if global_step % save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    policy_pipe.transformer.save_pretrained(str(ckpt_dir))
                    print(f"  Saved checkpoint: {ckpt_dir}")

    # Save final checkpoint
    final_dir = output_dir / "final"
    policy_pipe.transformer.save_pretrained(str(final_dir))
    print(f"\nFinal checkpoint saved to: {final_dir}")

    return global_step


def main():
    parser = argparse.ArgumentParser(description="Train SD3.5-M with DDRL (forward KL + reward)")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/ddrl")
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Limit prompts (useful for quick sanity checks)")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ddrl_cfg = config.get("ddrl", {})

    if not args.no_wandb:
        wandb.init(
            project=config.get("logging", {}).get("wandb", {}).get("project", "tfg-rl-baselines"),
            name="ddrl_training",
            config={"ddrl": ddrl_cfg, **config},
        )

    # Load prompts
    prompts_file = args.prompts_file or config.get("geneval", {}).get("prompts_file")
    if prompts_file and Path(prompts_file).exists():
        prompts = load_training_prompts(prompts_file, args.max_prompts)
    else:
        print("Warning: prompts file not found; using test prompts")
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
    )
    ref_pipe = setup_reference_model(model_name, device=device)

    detector_path = config.get("geneval", {}).get("detector_path", "repos/geneval/detector_models")
    reward_fn = setup_geneval_reward(detector_path)

    total_steps = train_ddrl(
        policy_pipe, ref_pipe, reward_fn, prompts,
        config, args.output_dir, device=device,
    )

    print(f"\nDDRL training complete. Total steps: {total_steps}")
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
