"""CPU test for GuidedSampler against a fake pi0-shaped model.

The fake copies openpi's ``sample_actions`` control flow verbatim (prefix cache,
``dt`` tensor, ``while time >= -dt / 2`` loop, ``x_t = x_t + dt * v_t``) over a
tiny random "network", so the test checks that the hook with rho=0 is bitwise
identical to the base loop and that rho>0 moves x0 toward a toy Q's optimum.

Run: python3 scripts/phase2/test_guided_sampler.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _make_att_2d_masks(pad_masks, att_masks):
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


# Stub the openpi module the sampler imports lazily.
_stub = types.ModuleType("openpi.models_pytorch.pi0_pytorch")
_stub.make_att_2d_masks = _make_att_2d_masks
sys.modules.setdefault("openpi", types.ModuleType("openpi"))
sys.modules.setdefault("openpi.models_pytorch", types.ModuleType("openpi.models_pytorch"))
sys.modules["openpi.models_pytorch.pi0_pytorch"] = _stub

from guided_sampler import GuidanceConfig, GuidedSampler, install  # noqa: E402

H, A, S, D, P = 10, 32, 32, 16, 6  # horizon, action dim, state dim, embed dim, prefix tokens


class _Expert:
    def __init__(self, g):
        self.paligemma = SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa")))
        self.gemma_expert = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa")))
        self.W = torch.randn(D, D, generator=g) / D**0.5

    def forward(self, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, adarms_cond=None):
        if past_key_values is None:  # prefix pass: cache = pooled prefix
            pe = inputs_embeds[0]
            return None, (pe.mean(1) @ self.W)
        se = inputs_embeds[1]
        kv = past_key_values
        out = torch.tanh(se @ self.W + kv[:, None, :])
        return [None, out], None


class FakePi0:
    """Mimics the subset of PI0Pytorch that sample_actions touches."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.config = SimpleNamespace(action_horizon=H, action_dim=A)
        self.paligemma_with_expert = _Expert(g)
        self.Win = torch.randn(A + S + 1, D, generator=g) / (A + S + 1) ** 0.5
        self.Wout = torch.randn(D, A, generator=g) / D**0.5
        self.pos = torch.randn(H, D, generator=g) * 0.1

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def _preprocess_observation(self, observation, *, train=True):
        return observation.images, observation.img_masks, observation.lang, observation.lang_masks, observation.state

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        B = images.shape[0]
        embs = torch.cat([images, lang_tokens], dim=1)
        pad = torch.ones(B, P, dtype=torch.bool)
        att = torch.zeros(B, P, dtype=torch.long)
        return embs, pad, att

    def _prepare_attention_masks_4d(self, m):
        return m[:, None]

    def embed_suffix(self, state, noisy_actions, timestep):
        B = state.shape[0]
        t = timestep.view(B, 1, 1).expand(B, H, 1)
        x = torch.cat([noisy_actions, state[:, None, :].expand(B, H, S), t], dim=-1)
        return x @ self.Win + self.pos, torch.ones(B, H, dtype=torch.bool), torch.ones(B, H, dtype=torch.long), None

    def action_out_proj(self, h):
        return h @ self.Wout

    # ---- copied from openpi pi0_pytorch.py (sample_actions / denoise_step) ----
    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True)
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d, position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs], use_cache=False, adarms_cond=[None, adarms_cond])
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


def _obs(B, g):
    return SimpleNamespace(
        images=torch.randn(B, P - 2, D, generator=g), img_masks=None,
        lang=torch.randn(B, 2, D, generator=g), lang_masks=None,
        state=torch.randn(B, S, generator=g),
    )


def main():
    g = torch.Generator().manual_seed(1)
    model = FakePi0()
    obs = _obs(2, g)
    noise = torch.randn(2, H, A, generator=g)
    dev = torch.device("cpu")

    # 1. rho=0 hook is bitwise identical to the base loop, for both step counts.
    for ns in (5, 10):
        ref = model.sample_actions(dev, obs, noise=noise, num_steps=ns)
        hook = GuidedSampler(model, q_fn=None, cfg=GuidanceConfig(rho=0.0))
        out = hook(dev, obs, noise=noise, num_steps=ns)
        assert torch.equal(ref, out), f"rho=0 mismatch, num_steps={ns}, max diff {(ref-out).abs().max()}"
        # also with a q_fn attached but rho=0, and with recording on
        hook = GuidedSampler(model, q_fn=lambda c, x: -(x**2).mean((1, 2)), cfg=GuidanceConfig(rho=0.0), record=True)
        out = hook(dev, obs, noise=noise, num_steps=ns)
        assert torch.equal(ref, out), "rho=0 with q_fn attached must not change the result"
        r = hook.records[-1]
        assert len(r.x0_hats) == ns and r.actions.shape == (2, H, A) and r.prefix_pool.shape == (2, D) and r.state.shape == (2, S)
    print("ok: rho=0 bitwise identical (num_steps 5, 10); records well-formed")

    # 2. built-in equivalence self-check path.
    hook = GuidedSampler(model, q_fn=None, equivalence_checks=3, record=True)
    for _ in range(4):
        hook(dev, obs, noise=noise, num_steps=5)
    assert hook.equivalence_results == [0.0, 0.0, 0.0] and hook.records[0].equivalence_max_abs_diff == 0.0
    assert hook.records[3].equivalence_max_abs_diff is None
    print("ok: equivalence self-check runs for the first N calls only, reports 0.0")

    # 3. guidance toward a toy Q moves x0 monotonically with rho and logs Q.
    target = torch.zeros(1, H, A)
    q_fn = lambda ctx, x: -((x - target) ** 2).flatten(1).sum(1)  # noqa: E731
    base = model.sample_actions(dev, obs, noise=noise, num_steps=5)
    d_prev = float(((base - target) ** 2).mean())
    # Normalized gradient steps have fixed length rho*sqrt(numel); large rho overshoots a quadratic bowl.
    for rho in (0.02, 0.05, 0.1):
        hook = GuidedSampler(model, q_fn=q_fn, cfg=GuidanceConfig(rho=rho), record=True)
        out = hook(dev, obs, noise=noise, num_steps=5)
        d = float(((out - target) ** 2).mean())
        assert d < d_prev, f"rho={rho}: distance to Q optimum {d:.4f} not below {d_prev:.4f}"
        assert len(hook.records[-1].q_values) == 5
        d_prev = d
    print("ok: larger rho moves the sample closer to the toy Q optimum; Q logged per fired step")

    # 4. apply_range gates firing; grad_steps>1 is a stronger push.
    hook_all = GuidedSampler(model, q_fn=q_fn, cfg=GuidanceConfig(rho=0.1), record=True)
    hook_late = GuidedSampler(model, q_fn=q_fn, cfg=GuidanceConfig(rho=0.1, apply_range=(0.0, 0.5)), record=True)
    hook_all(dev, obs, noise=noise, num_steps=5); hook_late(dev, obs, noise=noise, num_steps=5)
    assert len(hook_all.records[-1].q_values) == 5 and len(hook_late.records[-1].q_values) == 2
    out2 = GuidedSampler(model, q_fn=q_fn, cfg=GuidanceConfig(rho=0.1, grad_steps=3))(dev, obs, noise=noise, num_steps=5)
    assert float(((out2 - target) ** 2).mean()) < float(((hook_all.records[-1].actions - target) ** 2).mean())
    print("ok: apply_range gates firing (5 vs 2 fires); grad_steps=3 pushes further")

    # 4b. a torch.compile-style wrapper on sample_actions: the strict check uses the original callable,
    #     the compiled one is reported separately.
    class Wrapped(FakePi0):
        pass
    wm = Wrapped()
    orig = wm.sample_actions
    def fake_compiled(device, observation, noise=None, num_steps=10):
        return orig(device, observation, noise=noise, num_steps=num_steps) + 0.01  # "different numerics"
    fake_compiled._torchdynamo_orig_callable = orig
    wm.sample_actions = fake_compiled
    hook = GuidedSampler(wm, equivalence_checks=2)
    for _ in range(2):
        hook(dev, obs, noise=noise, num_steps=5)
    assert hook.equivalence_results == [0.0, 0.0], hook.equivalence_results
    assert len(hook.compiled_diffs) == 2 and abs(hook.compiled_diffs[0][1] - 0.01) < 1e-6
    print("ok: compiled wrapper detected; strict check against eager original, compiled diff reported")

    # 5. install() routes a fake Policy.
    pol = SimpleNamespace(_is_pytorch_model=True, _sample_actions=model.sample_actions)
    s = GuidedSampler(model)
    install(pol, s)
    assert pol._sample_actions is s
    print("ok: install() replaces Policy._sample_actions")
    print("ALL PASS")


if __name__ == "__main__":
    main()
