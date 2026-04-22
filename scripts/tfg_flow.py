"""
TFG-Flow: gradient-based training-free guidance for SD3 flow-matching.

Adapts the algorithm from Ye et al., "Training-Free Guidance via Score Diffusion"
(methods/tfg.py in YWolfeee/Training-Free-Guidance) to FlowMatchEulerDiscreteScheduler.

Conventions (verified against diffusers FlowMatchEulerDiscreteScheduler.step):
    x_t        = sigma * eps + (1 - sigma) * x_0
    v          = eps - x_0                              (model output)
    x_0_pred   = x_t - sigma * v
    prev_step  = x_t + dt * v,   dt = sigma_next - sigma  (dt < 0)

The hook runs before the scheduler step. Given current (x_t, v_detached, sigma),
we re-run the transformer inside a grad scope on a clone of x_t so that the
gradient flows through velocity -> x_0 -> VAE decode -> predictor. Without the
re-run, x_0_pred would be an affine function of x_t and the guidance degenerates
to pure classifier guidance on a linear projection (cheaper but weaker).

For cost control, guidance fires every `apply_every_n_steps` calls by default.
At ~40 inference steps this gives 10 guided steps, each roughly 2x the cost of
an unguided step (transformer re-run + VAE decode + predictor + backward).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PromptContext:
    """Per-call prompt state the guidance hook needs to re-run the transformer.

    encode_prompt output is reused so the hook doesn't re-tokenize every step.
    prompt_text is kept separately for the predictor (CLIP wants raw text).
    """

    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    prompt_text: list[str]


class TFGFlowGuidance:
    def __init__(
        self,
        predictor,
        guidance_scale: float = 1.0,
        recurrence_steps: int = 1,
        apply_every_n_steps: int = 4,
        apply_range: tuple[float, float] = (0.2, 0.8),
        clip_scale: float = 1.0,
    ):
        """
        predictor: callable(image[B,3,H,W] in [-1,1], prompt_text: list[str]) -> score[B]
                   Score is higher-is-better; differentiable wrt image.
        guidance_scale (rho): strength of the gradient update applied to x_t.
        recurrence_steps: number of refine-renoise cycles per diffusion step.
                          Upstream TFG uses this to stabilize the gradient; we keep the
                          knob but default to 1 since recurrence ~doubles cost per fire.
        apply_every_n_steps: only fire on every N-th scheduler step. Cheapest axis
                             for controlling TFG cost. N=1 is every step.
        apply_range: (low, high) fraction of the denoising trajectory (by sigma) in
                     which to fire guidance. (0.2, 0.8) skips earliest and latest steps
                     where guidance is either too noisy (early) or too committed (late).
        clip_scale: per-sample gradient rescale target (||g|| ≈ clip_scale * sqrt(numel)).
        """
        self.predictor = predictor
        self.rho = float(guidance_scale)
        self.recurrence_steps = int(recurrence_steps)
        self.apply_every_n_steps = int(apply_every_n_steps)
        self.apply_range = apply_range
        self.clip_scale = float(clip_scale)

        self._prompt_context: PromptContext | None = None
        self._step_counter = 0
        self._fires = 0

    def set_prompt_context(self, ctx: PromptContext) -> None:
        """Call once per pipe(...) invocation, before denoising starts.

        Resets the per-run step counter so apply_every_n_steps aligns to the new
        trajectory — otherwise two back-to-back generations would have offset fires.
        """
        self._prompt_context = ctx
        self._step_counter = 0
        self._fires = 0

    @property
    def fires_this_run(self) -> int:
        """Diagnostic: how many times guidance actually fired in the last run.

        Lets callers assert that TFG did something rather than silently skipping.
        """
        return self._fires

    def apply(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        pipe,
    ) -> torch.Tensor:
        """Modify x_t in place of the scheduler's input sample.

        Called from the patched scheduler.step, before the Euler update runs.
        Returns the possibly-modified sample; scheduler does its normal math on it.
        """
        step_idx = self._step_counter
        self._step_counter += 1

        if self._prompt_context is None:
            raise RuntimeError(
                "TFGFlowGuidance.apply called without set_prompt_context. "
                "Call set_prompt_context() before pipe(prompt, ...)."
            )

        if step_idx % self.apply_every_n_steps != 0:
            return sample

        sigma = self._current_sigma(pipe.scheduler, timestep, sample)
        sigma_scalar = float(sigma.flatten()[0].item())
        low, high = self.apply_range
        if not (low <= sigma_scalar <= high):
            return sample

        ctx = self._prompt_context
        transformer = pipe.transformer
        vae = pipe.vae

        # With enable_model_cpu_offload, transformer/vae parameters may be on CPU
        # at the moment we enter this hook — the accelerate wrapper moves them
        # to the configured device on forward(). The forward-time target is
        # pipe._execution_device (accelerate) or, as a safer fallback, the
        # device of the `sample` tensor (always on the execution device during
        # the scheduler's denoising loop). Use that for ALL our inputs so the
        # accelerate hook pre-moves the module correctly.
        exec_dev = sample.device

        guided_sample = sample
        for _ in range(self.recurrence_steps):
            with torch.enable_grad():
                x_g = guided_sample.detach().to(device=exec_dev, dtype=torch.float32).requires_grad_(True)

                t_in = timestep.expand(x_g.shape[0]) if timestep.dim() == 0 else timestep
                v_g = transformer(
                    hidden_states=x_g.to(device=exec_dev, dtype=transformer.dtype),
                    timestep=t_in.to(exec_dev),
                    encoder_hidden_states=ctx.prompt_embeds.to(device=exec_dev, dtype=transformer.dtype),
                    pooled_projections=ctx.pooled_prompt_embeds.to(device=exec_dev, dtype=transformer.dtype),
                    return_dict=False,
                )[0].to(torch.float32)

                x0_pred = x_g - sigma.to(exec_dev) * v_g

                # SD3 VAE: encode returns (latent - shift) * scale, so decode reverses that.
                x0_for_decode = x0_pred / vae.config.scaling_factor + vae.config.shift_factor
                image = vae.decode(
                    x0_for_decode.to(device=exec_dev, dtype=vae.dtype), return_dict=False
                )[0].to(torch.float32)

                score = self.predictor(image, ctx.prompt_text)
                if score.dim() == 0:
                    score = score.unsqueeze(0)
                grad_x = torch.autograd.grad(score.sum(), x_g, create_graph=False)[0]
                grad_x = self._rescale(grad_x)

            # Gradient points in the direction that INCREASES the predictor score.
            # We want to increase score, so step with +rho * grad.
            guided_sample = guided_sample + self.rho * grad_x.to(guided_sample.dtype)

        self._fires += 1
        return guided_sample.detach()

    def _current_sigma(
        self, scheduler, timestep: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        """Return sigma for the current step, broadcastable to sample shape.

        FlowMatchEuler stores sigmas[] indexed by step_index (set inside step).
        If step_index is set, use it; otherwise fall back to timestep/num_train.
        """
        if getattr(scheduler, "step_index", None) is not None:
            sigma = scheduler.sigmas[scheduler.step_index].to(sample.device, sample.dtype)
        else:
            num_train = getattr(
                getattr(scheduler, "config", None), "num_train_timesteps", 1000
            )
            sigma = (timestep.to(sample.dtype) / num_train).to(sample.device)
        # Broadcast to (B, 1, 1, 1)
        while sigma.dim() < sample.dim():
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _rescale(self, grad: torch.Tensor) -> torch.Tensor:
        """Per-sample gradient norm clip.

        Matches upstream TFG's `rescale_grad`: keep ||g|| ≤ clip_scale * sqrt(numel).
        Without this, the first few guided steps can blow up due to large gradient
        norms near the decision boundary of the predictor.
        """
        b = grad.shape[0]
        flat = grad.reshape(b, -1)
        numel = flat.shape[-1]
        target = self.clip_scale * math.sqrt(numel)
        norm = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (target / norm).clamp(max=1.0)
        return (flat * scale).reshape(grad.shape)


def patch_scheduler_step(pipe, guidance: TFGFlowGuidance) -> None:
    """Wrap pipe.scheduler.step so guidance runs on sample before the real step.

    Idempotent-ish: stores the original step on the guidance object so calling
    this twice doesn't double-wrap. Caller is responsible for not mixing hooks.
    """
    if hasattr(pipe.scheduler, "_tfg_original_step"):
        return  # already patched
    original_step = pipe.scheduler.step
    pipe.scheduler._tfg_original_step = original_step

    def tfg_step(model_output, timestep, sample, *args, **kwargs):
        guided = guidance.apply(model_output, timestep, sample, pipe)
        return original_step(model_output, timestep, guided, *args, **kwargs)

    pipe.scheduler.step = tfg_step


def unpatch_scheduler_step(pipe) -> None:
    """Restore original scheduler.step. Safe to call if not patched."""
    original = getattr(pipe.scheduler, "_tfg_original_step", None)
    if original is not None:
        pipe.scheduler.step = original
        delattr(pipe.scheduler, "_tfg_original_step")
