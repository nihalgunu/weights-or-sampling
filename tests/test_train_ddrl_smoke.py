"""
CPU smoke test for train_ddrl internals.

Verifies shape plumbing and gradient flow through the rewritten training loop
helpers (`rollout`, `decode_and_score`, `compute_velocity_kl`) using stubs in
place of SD3's transformer/VAE and CLIP. Catches things like: velocity shape
mismatches, sigma indexing off-by-one, grad-scope partitioning bugs, KL term
losing grad, optimizer receiving empty parameter lists.

Does NOT verify: real SD3 shapes (16 latent channels, 4096-dim embed), actual
LoRA injection, mmdetection anything, or that the loss decreases. Those need a
GPU + diffusers + peft and are caught on Lambda.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_ddrl import (  # noqa: E402
    compute_velocity_kl,
    decode_and_score,
    rollout,
)


# ─────────────────────────────────────────────────────────────────────────────
# Stubs
# ─────────────────────────────────────────────────────────────────────────────


class StubTransformer(nn.Module):
    """Small conv mimicking SD3 transformer signature. Returns (velocity,)."""

    def __init__(self, channels=4):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.dtype = torch.float32
        self.config = SimpleNamespace(in_channels=channels)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        pooled_projections,
        return_dict=False,
    ):
        del timestep, encoder_hidden_states, pooled_projections
        v = self.conv(hidden_states)
        if return_dict:
            return SimpleNamespace(sample=v)
        return (v,)


class StubVAE(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, 3, padding=1)
        self.dtype = torch.float32
        self.config = SimpleNamespace(scaling_factor=1.5305, shift_factor=0.0609)

    def decode(self, z, return_dict=False):
        img = torch.tanh(self.conv(z))
        if return_dict:
            return SimpleNamespace(sample=img)
        return (img,)


class StubPredictor:
    """Differentiable scalar: mean of image pixels. Shape: (B,)."""

    def __call__(self, image, prompt_text):
        del prompt_text
        return image.view(image.shape[0], -1).mean(dim=-1)


class StubScheduler:
    """Mimics FlowMatchEulerDiscrete enough for rollout() to run."""

    def __init__(self, num_steps=4):
        # Decreasing sigmas from 1.0 to 0.0 (inclusive endpoints).
        self.sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
        self.timesteps = (self.sigmas[:-1] * 1000.0)
        self.config = SimpleNamespace(num_train_timesteps=1000)

    def set_timesteps(self, num_inference_steps, device=None):
        # Stub ignores num_inference_steps — we set it in __init__.
        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.sigmas = self.sigmas.to(device)


def _make_stub_pipe(batch=2, channels=4, spatial=8):
    pipe = SimpleNamespace(
        transformer=StubTransformer(channels=channels),
        vae=StubVAE(in_channels=channels),
        scheduler=StubScheduler(num_steps=4),
        vae_scale_factor=1,  # stub uses 1-to-1 spatial; real SD3 uses 8
    )
    pipe._shape = (batch, channels, spatial, spatial)
    return pipe


def _fake_embeds(B, seq=77, dim=16, pooled_dim=8):
    """Arbitrary-dim embeds — stubs don't care about 4096/2048."""
    return (
        torch.randn(B, seq, dim),
        torch.randn(B, seq, dim),
        torch.randn(B, pooled_dim),
        torch.randn(B, pooled_dim),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_rollout_returns_final_shape_and_flows_grad_through_last_step():
    pipe = _make_stub_pipe(batch=2, channels=4, spatial=8)
    B, C, H, W = pipe._shape
    pos, neg, pos_p, neg_p = _fake_embeds(B)
    gen = torch.Generator().manual_seed(0)

    latents = rollout(
        pipe, pos, pos_p, neg, neg_p,
        latent_shape=(B, C, H, W),
        num_inference_steps=4,
        grad_steps=1,
        guidance_scale=4.5,
        generator=gen,
        device="cpu",
    )
    assert latents.shape == (B, C, H, W), f"rollout shape {latents.shape}"
    assert latents.requires_grad, "rollout final latents must carry grad (last step was grad-enabled)"


def test_rollout_grad_scope_zero_means_no_grad():
    pipe = _make_stub_pipe(batch=2)
    B, C, H, W = pipe._shape
    pos, neg, pos_p, neg_p = _fake_embeds(B)
    gen = torch.Generator().manual_seed(1)

    latents = rollout(
        pipe, pos, pos_p, neg, neg_p,
        latent_shape=(B, C, H, W),
        num_inference_steps=4,
        grad_steps=0,
        guidance_scale=4.5,
        generator=gen,
        device="cpu",
    )
    assert not latents.requires_grad, "grad_steps=0 should return detached latents"


def test_rollout_without_cfg_runs():
    pipe = _make_stub_pipe(batch=2)
    B, C, H, W = pipe._shape
    pos, neg, pos_p, neg_p = _fake_embeds(B)
    gen = torch.Generator().manual_seed(2)

    # guidance_scale <= 1.0 -> skip the CFG path; single transformer forward.
    latents = rollout(
        pipe, pos, pos_p, neg, neg_p,
        latent_shape=(B, C, H, W),
        num_inference_steps=4,
        grad_steps=1,
        guidance_scale=1.0,
        generator=gen,
        device="cpu",
    )
    assert latents.shape == (B, C, H, W)


def test_decode_and_score_gives_batch_scalar_with_grad():
    pipe = _make_stub_pipe(batch=3)
    B, C, H, W = pipe._shape
    latents = torch.randn(B, C, H, W, requires_grad=True)
    score = decode_and_score(pipe.vae, StubPredictor(), latents, ["a", "b", "c"])
    assert score.shape == (B,), f"score shape {score.shape}"
    # Backward must flow to latents (the only leaf with requires_grad).
    score.sum().backward()
    assert latents.grad is not None, "decode_and_score broke the grad path"
    assert latents.grad.abs().sum().item() > 0, "zero grad through predictor?"


def test_velocity_kl_is_scalar_and_has_policy_grad_only():
    torch.manual_seed(3)
    policy = StubTransformer(channels=4)
    ref = StubTransformer(channels=4)
    for p in ref.parameters():
        p.requires_grad_(False)

    B, C, H, W = 2, 4, 8, 8
    x0 = torch.randn(B, C, H, W)  # detached
    pos_e, _, pos_p, _ = _fake_embeds(B)
    scheduler = StubScheduler(num_steps=4)

    kl = compute_velocity_kl(policy, ref, x0, pos_e, pos_p, scheduler, device="cpu")
    assert kl.dim() == 0, f"KL should be scalar, got shape {kl.shape}"
    assert kl.item() >= 0.0, f"KL (MSE) must be non-negative, got {kl.item()}"

    # Gradient must flow into policy params but not into ref params.
    kl.backward()
    policy_grad_norm = sum(p.grad.abs().sum().item() for p in policy.parameters() if p.grad is not None)
    assert policy_grad_norm > 0, "policy did not receive gradient from KL term"
    for p in ref.parameters():
        assert p.grad is None, "ref params received gradient — KL is not properly detached"


def test_full_training_step_loss_backward_updates_policy_only():
    """End-to-end: rollout -> reward -> KL -> loss -> backward updates policy conv weights.

    This is the integration check: if the rewrite is plumbed correctly, a single
    gradient step should leave policy.conv.weight.grad non-zero and ref conv.grad None.
    """
    torch.manual_seed(4)
    pipe = _make_stub_pipe(batch=2)
    B, C, H, W = pipe._shape
    ref_pipe = _make_stub_pipe(batch=2)
    for p in ref_pipe.transformer.parameters():
        p.requires_grad_(False)

    pos, neg, pos_p, neg_p = _fake_embeds(B)
    predictor = StubPredictor()
    gen = torch.Generator().manual_seed(5)

    latents = rollout(
        pipe, pos, pos_p, neg, neg_p,
        latent_shape=(B, C, H, W),
        num_inference_steps=4,
        grad_steps=1,
        guidance_scale=4.5,
        generator=gen,
        device="cpu",
    )
    rewards = decode_and_score(pipe.vae, predictor, latents, ["x", "y"])
    reward_loss = -rewards.mean()

    kl = compute_velocity_kl(
        pipe.transformer, ref_pipe.transformer,
        latents.detach(), pos, pos_p, pipe.scheduler, device="cpu",
    )
    loss = reward_loss + 0.05 * kl

    assert torch.isfinite(loss), f"loss is non-finite: {loss.item()}"
    loss.backward()

    policy_grad = pipe.transformer.conv.weight.grad
    assert policy_grad is not None and policy_grad.abs().sum().item() > 0, (
        "policy transformer did not receive gradient from combined loss"
    )
    for p in ref_pipe.transformer.parameters():
        assert p.grad is None, "ref transformer received gradient — leakage via KL path"


def test_can_overfit_single_batch():
    """Convergence sanity check: 30 AdamW steps on the SAME batch must reduce the loss.

    Classic sanity check that the training loop actually learns. If the model
    can't even memorize one batch (i.e., loss doesn't go down), something is
    fundamentally broken — LoRA not receiving gradients, optimizer not stepping,
    target signal not differentiable, etc. This catches silent training bugs
    before we commit to a multi-hour $$$ GPU training run.

    We use the same rollout + decode + score + velocity-KL pipeline as the real
    training loop, run it many times on one fixed noise batch, and assert that
    mean loss of the last 10 steps is strictly below mean loss of the first 10.
    """
    torch.manual_seed(7)
    pipe = _make_stub_pipe(batch=2)
    B, C, H, W = pipe._shape
    ref_pipe = _make_stub_pipe(batch=2)
    for p in ref_pipe.transformer.parameters():
        p.requires_grad_(False)

    # Copy ref weights to policy so they start identical — this makes the KL
    # term start at 0 and the loss is dominated by the reward term, which is
    # what actually needs to converge. Otherwise the KL regularizer dominates
    # and the policy can't move toward the reward.
    ref_pipe.transformer.load_state_dict(pipe.transformer.state_dict())

    pos, neg, pos_p, neg_p = _fake_embeds(B)
    predictor = StubPredictor()  # = mean(image) — model is incentivized to maximize mean pixel value

    # Small LR but high enough to converge in 30 steps on a 39-param conv.
    optimizer = torch.optim.AdamW(pipe.transformer.parameters(), lr=1e-2)

    losses = []
    # Fixed noise seed so every step rolls out from the same initial noise.
    for step in range(30):
        gen = torch.Generator().manual_seed(100)
        latents = rollout(
            pipe, pos, pos_p, neg, neg_p,
            latent_shape=(B, C, H, W),
            num_inference_steps=4,
            grad_steps=1,
            guidance_scale=4.5,
            generator=gen,
            device="cpu",
        )
        rewards = decode_and_score(pipe.vae, predictor, latents, ["x", "y"])
        reward_loss = -rewards.mean()
        # Keep KL small so reward dominates — we just want to verify the policy
        # can move in the reward direction at all.
        kl = compute_velocity_kl(
            pipe.transformer, ref_pipe.transformer,
            latents.detach(), pos, pos_p, pipe.scheduler, device="cpu",
        )
        loss = reward_loss + 0.01 * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first_mean = sum(losses[:10]) / 10
    last_mean = sum(losses[-10:]) / 10
    drop = first_mean - last_mean
    assert last_mean < first_mean, (
        f"Loss did NOT drop across 30 steps on a fixed batch — training loop is broken. "
        f"first_10_mean={first_mean:.4f}  last_10_mean={last_mean:.4f}"
    )
    # Drop should be non-trivial (> 1% of the starting magnitude) — otherwise
    # we're learning nothing. 1% is loose enough to tolerate stub noise while
    # catching real bugs (LR=0, no grad, etc.).
    rel_drop = drop / abs(first_mean) if first_mean != 0 else 0
    assert rel_drop > 0.01, (
        f"Loss drop too small to be real learning: first={first_mean:.4f}  "
        f"last={last_mean:.4f}  drop={drop:.4f}  rel_drop={rel_drop:.4f}"
    )
    print(f"  [convergence] first_10={first_mean:.4f}  last_10={last_mean:.4f}  "
          f"drop={drop:.4f}  rel_drop={rel_drop*100:.1f}%")


def _run_all():
    tests = [
        test_rollout_returns_final_shape_and_flows_grad_through_last_step,
        test_rollout_grad_scope_zero_means_no_grad,
        test_rollout_without_cfg_runs,
        test_decode_and_score_gives_batch_scalar_with_grad,
        test_velocity_kl_is_scalar_and_has_policy_grad_only,
        test_full_training_step_loss_backward_updates_policy_only,
        test_can_overfit_single_batch,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failures.append(t.__name__)
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
