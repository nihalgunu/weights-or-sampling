"""Training-free guidance hook for openpi's PyTorch pi0 / pi0.5 sampler.

Replaces ``PI0Pytorch.sample_actions`` (rlinf-openpi 0.1.1 == upstream openpi,
``src/openpi/models_pytorch/pi0_pytorch.py``) with a loop that is arithmetically
identical when no guidance fires, and that otherwise nudges the running sample
``x_t`` with the gradient of a learned value ``Q(ctx, x0_hat)`` taken at the
one-step clean-action estimate ``x0_hat = x_t - t * v_t``.

Flow convention (openpi ``forward``): ``x_t = t * eps + (1 - t) * x0`` and the
target velocity is ``u_t = eps - x0``; Euler integrates from ``t = 1`` with
``dt = -1 / num_steps``.

Install on a loaded ``openpi.policies.policy.Policy`` with
``install(policy, sampler)``; ``Policy.infer`` calls ``policy._sample_actions``.

The sampler never backpropagates through the policy network. Only ``Q`` sees a
gradient, so the per-step cost is one small forward+backward of ``Q``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

QFn = Callable[[dict[str, Any], torch.Tensor], torch.Tensor]  # (ctx, x0_hat[B,H,A]) -> value[B]


@dataclass
class GuidanceConfig:
    rho: float = 0.0  # strength; 0 disables guidance and reproduces the base sampler exactly
    apply_range: tuple[float, float] = (0.0, 1.0)  # fire only when t in [low, high] (t runs 1 -> 1/num_steps)
    clip_scale: float = 1.0  # per-sample gradient rescale: ||g|| = clip_scale * sqrt(numel(x0))
    grad_steps: int = 1  # ascent iterations on x0_hat per fired step
    normalize: bool = True  # rescale the gradient to the fixed norm above (False = raw gradient * rho)


@dataclass
class SampleRecord:
    """What one call to the sampler saw and produced (CPU, float16 where large)."""

    noise: torch.Tensor  # [B,H,A]
    state: torch.Tensor  # [B,S] normalized model state
    prefix_pool: torch.Tensor  # [B,D] masked mean of prefix input embeddings (image + language tokens)
    x0_hats: list[torch.Tensor] = field(default_factory=list)  # per step, [B,H,A], before guidance
    times: list[float] = field(default_factory=list)
    actions: Optional[torch.Tensor] = None  # final x_t [B,H,A] (normalized model space)
    q_values: list[float] = field(default_factory=list)  # Q(x0_hat) per fired step, batch mean
    equivalence_max_abs_diff: Optional[float] = None  # set when the self-check ran
    bon_q: list[float] = field(default_factory=list)  # best-of-N: Q of each candidate's final chunk
    bon_pick: int = 0  # index of the candidate returned


class GuidedSampler:
    """Drop-in for ``model.sample_actions(device, observation, noise=None, num_steps=10)``."""

    def __init__(
        self,
        model,
        q_fn: Optional[QFn] = None,
        cfg: Optional[GuidanceConfig] = None,
        *,
        record: bool = False,
        equivalence_checks: int = 0,
        bon: int = 1,
    ):
        self.model = model
        self.q_fn = q_fn
        self.cfg = cfg or GuidanceConfig()
        self.record = record
        self.bon = bon  # >1: rank that many candidate chunks (caller passes N noises) by q_fn, return the best
        self.records: list[SampleRecord] = []
        self.equivalence_checks = equivalence_checks
        self.equivalence_results: list[float] = []
        # openpi wraps model.sample_actions in torch.compile(mode="max-autotune"); the compiled graph's
        # numerics differ from eager, so the strict check compares against the original (eager) callable
        # and the compiled output is reported separately (max |diff| on the real action dims / all dims).
        # (not __wrapped__: @torch.no_grad leaves an unbound function there)
        self.ref_eager = getattr(model.sample_actions, "_torchdynamo_orig_callable", None)
        self.ref_compiled = model.sample_actions if self.ref_eager is not None else None
        if self.ref_eager is None:
            self.ref_eager = model.sample_actions
        self.compiled_diffs: list[tuple[float, float]] = []
        self.real_action_dims = 7
        self.calls = 0

    # -- helpers --------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.q_fn is not None and self.cfg.rho != 0.0

    def _fires(self, t: float) -> bool:
        lo, hi = self.cfg.apply_range
        return self.active and (lo - 1e-6) <= t <= (hi + 1e-6)

    def _guidance_delta(self, ctx: dict[str, Any], x0_hat: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Return (delta to add to x_t, mean Q at the start point)."""
        x = x0_hat.detach().float().clone()
        q0 = None
        delta = torch.zeros_like(x)
        for _ in range(max(1, self.cfg.grad_steps)):
            x.requires_grad_(True)
            with torch.enable_grad():
                q = self.q_fn(ctx, x)
                if q.ndim == 0:
                    q = q[None]
                g = torch.autograd.grad(q.sum(), x)[0]
            if q0 is None:
                q0 = float(q.detach().mean())
            g = g.detach()
            if self.cfg.normalize:
                n = g.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, *([1] * (g.ndim - 1)))
                g = g / n * (self.cfg.clip_scale * math.sqrt(g[0].numel()))
            step = self.cfg.rho * g
            delta = delta + step
            x = (x.detach() + step)
        return delta.to(x0_hat.dtype), q0 if q0 is not None else float("nan")

    # -- main entry -----------------------------------------------------------

    @torch.no_grad()
    def __call__(self, device, observation, noise=None, num_steps=10):
        m = self.model
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, m.config.action_horizon, m.config.action_dim)
            noise = m.sample_noise(actions_shape, device)

        ref = ref_c = None
        if self.calls < self.equivalence_checks:
            ref_noise = noise[:1] if self.bon > 1 else noise  # candidate 0 carries the reference's noise
            ref = self.ref_eager(device, observation, noise=ref_noise, num_steps=num_steps)
            if self.ref_compiled is not None:
                ref_c = self.ref_compiled(device, observation, noise=ref_noise, num_steps=num_steps)
        self.calls += 1

        images, img_masks, lang_tokens, lang_masks, state = m._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        # Same helper the base loop uses; imported lazily so the CPU test can stub the module.
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = m._prepare_attention_masks_4d(prefix_att_2d_masks)
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = m.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        ctx = {
            "state": state,
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
            "num_steps": num_steps,
        }

        # Best-of-N: the caller passes N noises for a batch of one; each candidate reuses the prefix cache.
        n_cand = 1
        if self.bon > 1:
            assert bsize == 1 and noise.shape[0] == self.bon, f"bon={self.bon} needs batch 1 and noise [{self.bon},H,A], got {tuple(noise.shape)}"
            assert self.q_fn is not None, "best-of-N needs a q_fn to rank candidates"
            n_cand = self.bon

        finals, traces = [], []
        for j in range(n_cand):
            x_j, trace = self._euler(noise[j : j + 1] if n_cand > 1 else noise, state, prefix_pad_masks, past_key_values,
                                     ctx, bsize, num_steps, device, keep_trace=self.record)
            finals.append(x_j); traces.append(trace)

        pick, bon_q = 0, []
        if n_cand > 1:
            with torch.no_grad():
                bon_q = [float(self.q_fn(ctx, x).reshape(-1)[0]) for x in finals]
            pick = int(torch.tensor(bon_q).argmax())
        x_t = finals[pick]

        if ref is not None:  # the check is against candidate 0, which shares the reference's noise
            d = float((ref.float() - finals[0].float()).abs().max())
            self.equivalence_results.append(d)
        if ref_c is not None:
            dc = (ref_c.float() - finals[0].float()).abs()
            self.compiled_diffs.append((float(dc[..., : self.real_action_dims].max()), float(dc.max())))
        if self.record:
            pm = prefix_pad_masks.to(prefix_embs.dtype)[..., None]
            pool = (prefix_embs.float() * pm).sum(1) / pm.sum(1).clamp_min(1.0)
            x0_hats, times, q_values = traces[pick]
            rec = SampleRecord(
                noise=(noise[pick : pick + 1] if n_cand > 1 else noise).detach().to("cpu", torch.float16),
                state=state.detach().to("cpu", torch.float32),
                prefix_pool=pool.detach().to("cpu", torch.float16),
                x0_hats=x0_hats, times=times, q_values=q_values,
                actions=x_t.detach().to("cpu", torch.float32),
                equivalence_max_abs_diff=self.equivalence_results[-1] if ref is not None else None,
                bon_q=bon_q, bon_pick=pick,
            )
            self.records.append(rec)
        return x_t

    def _euler(self, noise, state, prefix_pad_masks, past_key_values, ctx, bsize, num_steps, device, *, keep_trace):
        """The base Euler loop plus optional guidance; returns (x_t, (x0_hats, times, q_values))."""
        m = self.model
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x0_hats, times, q_values = [], [], []

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = m.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)

            t_val = float(time)
            if keep_trace or self._fires(t_val):
                x0_hat = x_t - time * v_t
                if keep_trace:
                    x0_hats.append(x0_hat.detach().to("cpu", torch.float16))
                    times.append(t_val)

            # Euler step, written exactly as the base sampler writes it.
            x_t = x_t + dt * v_t

            if self._fires(t_val):
                ctx["t"] = t_val
                delta, q0 = self._guidance_delta(ctx, x0_hat)
                x_t = x_t + delta
                q_values.append(q0)
            time += dt
        return x_t, (x0_hats, times, q_values)


def install(policy, sampler: GuidedSampler) -> None:
    """Route ``Policy.infer`` through ``sampler`` (openpi stores the bound method on the instance)."""
    assert getattr(policy, "_is_pytorch_model", False), "GuidedSampler needs the PyTorch policy"
    policy._sample_actions = sampler  # noqa: SLF001
