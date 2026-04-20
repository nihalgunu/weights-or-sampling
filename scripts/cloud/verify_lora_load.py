#!/usr/bin/env python3
"""
Pre-flight check: given a LoRA checkpoint path, verify that loading it
actually changes the SD3 transformer's output. ~45s of GPU.

Exists because pipe.load_lora_weights() on a PEFT-saved adapter for SD3 in
diffusers 0.37 is a silent no-op — we burned ~$4.50 on the H100 rediscovering
that. This script catches it in 45s before any long eval run.

Usage:
    python3 scripts/cloud/verify_lora_load.py <checkpoint_path>

Exits 0 if LoRA load affects output (diff > 1e-6); exits non-zero otherwise
so orchestrators can fail-fast.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to the PEFT-saved LoRA (dir with adapter_model.safetensors)")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--min_diff", type=float, default=1e-6,
                        help="Minimum mean-abs output difference to treat load as successful. "
                             "1e-6 is well above fp-noise and well below any real LoRA effect.")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[verify] loading {args.model}")
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    # Free VRAM — we don't need text encoders beyond the first prompt.
    if getattr(pipe, "text_encoder_3", None) is not None:
        pipe.text_encoder_3.to("cpu")
    torch.cuda.empty_cache()

    # One dummy prompt to get a valid embed shape for the transformer.
    with torch.no_grad():
        pos, _neg, pos_p, _neg_p = pipe.encode_prompt(
            prompt="a verification prompt",
            prompt_2=None, prompt_3=None,
            device="cuda", num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    # Temp move T5 back to CPU (encode_prompt may have pulled it to GPU).
    if getattr(pipe, "text_encoder_3", None) is not None:
        pipe.text_encoder_3.to("cpu")
    torch.cuda.empty_cache()

    in_ch = pipe.transformer.config.in_channels  # 16 for SD3.5-M
    torch.manual_seed(0)
    latent = torch.randn(1, in_ch, 64, 64, device="cuda", dtype=pipe.transformer.dtype)
    t = torch.tensor([500.0], device="cuda")

    pipe.transformer.eval()
    with torch.no_grad():
        out_base = pipe.transformer(
            hidden_states=latent,
            timestep=t,
            encoder_hidden_states=pos,
            pooled_projections=pos_p,
            return_dict=False,
        )[0].float()

    print(f"[verify] base output ready. mean abs = {out_base.abs().mean().item():.6e}")

    # Load LoRA using PEFT directly (diffusers' loader doesn't understand the
    # base_model.model.* prefix PEFT saves with).
    from peft import PeftModel
    print(f"[verify] loading LoRA from {args.checkpoint}")
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer, args.checkpoint, is_trainable=False
    )
    pipe.transformer.eval()

    with torch.no_grad():
        out_lora = pipe.transformer(
            hidden_states=latent,
            timestep=t,
            encoder_hidden_states=pos,
            pooled_projections=pos_p,
            return_dict=False,
        )[0].float()

    diff = (out_base - out_lora).abs().mean().item()
    max_diff = (out_base - out_lora).abs().max().item()
    elapsed = time.time() - t0
    print(f"[verify] diff mean-abs = {diff:.6e}   max-abs = {max_diff:.6e}   ({elapsed:.0f}s)")

    if diff < args.min_diff:
        print(f"[verify] FAIL — LoRA had no measurable effect. load path is broken.")
        sys.exit(1)

    # Also verify some lora_B weights are non-zero (belt + suspenders).
    b_norms = [
        p.float().abs().max().item()
        for n, p in pipe.transformer.named_parameters()
        if "lora_B" in n
    ]
    assert b_norms, "no lora_B parameters found — adapter did not attach"
    max_b = max(b_norms)
    print(f"[verify] max |lora_B| = {max_b:.4e}  across {len(b_norms)} B matrices")
    if max_b < 1e-8:
        print(f"[verify] FAIL — all lora_B weights are ~zero. Training didn't move them.")
        sys.exit(1)

    print("[verify] LoRA LOAD VERIFIED — downstream eval is safe to run")
    sys.exit(0)


if __name__ == "__main__":
    main()
