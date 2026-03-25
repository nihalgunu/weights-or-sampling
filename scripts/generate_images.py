#!/usr/bin/env python3
"""
Unified image generation script for TFG + RL experiments.
Supports:
- Base SD3.5-M inference
- TFG-augmented inference
- FlowGRPO checkpoint loading
- W&B image logging
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import wandb
import yaml
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prompts(prompts_file: str) -> list:
    """Load GenEval prompts."""
    # Try different prompt file formats
    if prompts_file.endswith('.json'):
        with open(prompts_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'prompts' in data:
                return data['prompts']
    elif prompts_file.endswith('.jsonl'):
        prompts = []
        with open(prompts_file, 'r') as f:
            for line in f:
                prompts.append(json.loads(line))
        return prompts

    # Fallback: try to find prompts in geneval repo
    geneval_prompts = PROJECT_ROOT / "repos/geneval/prompts"
    if geneval_prompts.exists():
        for f in geneval_prompts.glob("*.json*"):
            return load_prompts(str(f))

    raise FileNotFoundError(f"Could not load prompts from {prompts_file}")


def setup_model(model_name: str, checkpoint_path: str = None, device: str = "cuda"):
    """Initialize the diffusion model pipeline."""
    from diffusers import StableDiffusion3Pipeline

    print(f"Loading model: {model_name}")

    # Load base model
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    # Load finetuned checkpoint if provided
    if checkpoint_path and checkpoint_path != "null":
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from: {checkpoint_path}")

            # Check if it's a LoRA checkpoint
            lora_path = checkpoint_path / "adapter_model.safetensors"
            if lora_path.exists() or (checkpoint_path / "adapter_model.bin").exists():
                print("Loading LoRA weights...")
                pipe.load_lora_weights(str(checkpoint_path))
            else:
                # Full model checkpoint
                print("Loading full model checkpoint...")
                # Load transformer weights
                transformer_path = checkpoint_path / "transformer"
                if transformer_path.exists():
                    from diffusers import SD3Transformer2DModel
                    pipe.transformer = SD3Transformer2DModel.from_pretrained(
                        str(transformer_path),
                        torch_dtype=torch.bfloat16
                    )
        else:
            print(f"Warning: Checkpoint path {checkpoint_path} does not exist, using base model")

    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()  # Reduce VRAM usage

    return pipe


def setup_tfg(pipe, guidance_scale: float = 7.5, recurrence_steps: int = 1):
    """
    Setup TFG (Training-Free Guidance) for the pipeline.

    Note: This is a simplified implementation. For full TFG functionality,
    you may need to integrate with the TFG repo's implementation.
    """
    # TFG modifies the sampling process
    # For SD3/flow matching, TFG-Flow is needed

    # Store TFG parameters on the pipeline
    pipe.tfg_enabled = True
    pipe.tfg_guidance_scale = guidance_scale
    pipe.tfg_recurrence_steps = recurrence_steps

    print(f"TFG enabled: guidance_scale={guidance_scale}, recurrence_steps={recurrence_steps}")

    return pipe


def generate_images(
    pipe,
    prompts: list,
    output_dir: str,
    num_inference_steps: int = 40,
    guidance_scale: float = 4.5,
    seed: int = 42,
    use_tfg: bool = False,
) -> dict:
    """Generate images for all prompts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    results = {
        "prompts": [],
        "image_paths": [],
        "generation_times": [],
    }

    print(f"Generating {len(prompts)} images...")

    for i, prompt_data in enumerate(tqdm(prompts)):
        # Handle different prompt formats
        if isinstance(prompt_data, dict):
            prompt = prompt_data.get('prompt', prompt_data.get('text', str(prompt_data)))
            prompt_id = prompt_data.get('id', i)
        else:
            prompt = str(prompt_data)
            prompt_id = i

        start_time = time.time()

        # Generate image
        if use_tfg and hasattr(pipe, 'tfg_enabled') and pipe.tfg_enabled:
            # TFG-enhanced generation
            # Note: Full TFG implementation requires modifying the sampling loop
            # This is a simplified version using higher guidance scale
            image = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=pipe.tfg_guidance_scale,
                generator=generator,
            ).images[0]
        else:
            # Standard generation
            image = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

        gen_time = time.time() - start_time

        # Save image
        image_path = output_dir / f"{prompt_id:05d}.png"
        image.save(image_path)

        results["prompts"].append(prompt)
        results["image_paths"].append(str(image_path))
        results["generation_times"].append(gen_time)

        # Log to W&B periodically
        if wandb.run and i % 50 == 0:
            wandb.log({
                "generated_images": wandb.Image(image, caption=prompt[:100]),
                "generation_time": gen_time,
                "progress": i / len(prompts),
            })

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)

    avg_time = sum(results["generation_times"]) / len(results["generation_times"])
    print(f"Average generation time: {avg_time:.2f}s per image")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate images for TFG+RL experiments")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to config file")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Name of the run (e.g., run1_baseline)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for images")
    parser.add_argument("--use_tfg", type=str, default="false",
                        help="Whether to use TFG guidance")
    parser.add_argument("--tfg_guidance_scale", type=float, default=7.5,
                        help="TFG guidance scale")
    parser.add_argument("--tfg_recurrence_steps", type=int, default=1,
                        help="TFG recurrence steps")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to finetuned checkpoint")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Path to prompts file (overrides config)")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Number of images to generate (for testing)")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Parse boolean
    use_tfg = args.use_tfg.lower() in ('true', '1', 'yes')

    # Initialize W&B
    wandb.init(
        project=config.get('logging', {}).get('wandb', {}).get('project', 'tfg-rl-baselines'),
        name=args.run_name,
        config={
            "run_name": args.run_name,
            "use_tfg": use_tfg,
            "checkpoint": args.checkpoint,
            **config
        }
    )

    # Load prompts
    prompts_file = args.prompts_file or config.get('geneval', {}).get('prompts_file')
    if prompts_file:
        prompts = load_prompts(prompts_file)
    else:
        # Fallback: use some test prompts
        prompts = [
            {"id": 0, "prompt": "A red apple on a wooden table"},
            {"id": 1, "prompt": "Two cats sitting on a sofa"},
            {"id": 2, "prompt": "A blue car parked next to a green tree"},
        ]
        print("Warning: Using test prompts. Set prompts_file for full evaluation.")

    # Limit prompts if specified
    if args.num_images:
        prompts = prompts[:args.num_images]

    # Setup model
    model_name = config.get('model', {}).get('name', 'stabilityai/stable-diffusion-3.5-medium')
    pipe = setup_model(model_name, args.checkpoint)

    # Setup TFG if enabled
    if use_tfg:
        pipe = setup_tfg(pipe, args.tfg_guidance_scale, args.tfg_recurrence_steps)

    # Generate images
    gen_config = config.get('generation', {})
    results = generate_images(
        pipe,
        prompts,
        args.output_dir,
        num_inference_steps=gen_config.get('num_inference_steps', 40),
        guidance_scale=gen_config.get('guidance_scale', 4.5),
        seed=gen_config.get('seed', 42),
        use_tfg=use_tfg,
    )

    # Log final stats
    wandb.log({
        "total_images": len(results["image_paths"]),
        "avg_generation_time": sum(results["generation_times"]) / len(results["generation_times"]),
    })

    wandb.finish()
    print(f"Generation complete. Images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
