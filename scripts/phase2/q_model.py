"""Success-value model Q(obs, action chunk) for guidance and best-of-N over pi0.5 samples.

Inputs come from ``GuidedSampler`` records / ``libero_eval_guided.py`` dumps:
  state       [B, 32]  normalized model state (LIBERO uses the first 8 dims)
  prefix_pool [B, D]   masked mean of the prefix input embeddings (image + language tokens)
  x0          [B, H, A] normalized action chunk (LIBERO uses the first 7 of A=32 dims)
Output: logit of P(episode success | obs, chunk). Trained with BCE on the episode label
(Monte Carlo return, gamma = 1) attached to every inference of the episode.

``load_q(path, device)`` returns a callable ``(ctx, x0) -> [B]`` for ``GuidedSampler``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

STATE_DIMS = 8
ACT_DIMS = 7


def pool_prefix(prefix_embs: torch.Tensor, prefix_pad_masks: torch.Tensor) -> torch.Tensor:
    pm = prefix_pad_masks.to(prefix_embs.dtype)[..., None]
    return (prefix_embs.float() * pm).sum(1) / pm.sum(1).clamp_min(1.0)


class QNet(nn.Module):
    def __init__(self, prefix_dim: int, horizon: int, *, proj: int = 128, hidden: int = 256, dropout: float = 0.1,
                 use_prefix: bool = True):
        super().__init__()
        self.prefix_dim, self.horizon, self.use_prefix = prefix_dim, horizon, use_prefix
        self.arch = {"prefix_dim": prefix_dim, "horizon": horizon, "proj": proj, "hidden": hidden,
                     "dropout": dropout, "use_prefix": use_prefix}
        self.prefix_proj = nn.Sequential(nn.LayerNorm(prefix_dim), nn.Linear(prefix_dim, proj), nn.GELU()) if use_prefix else None
        in_dim = STATE_DIMS + horizon * ACT_DIMS + (proj if use_prefix else 0)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        # feature standardization filled by the trainer
        self.register_buffer("state_mu", torch.zeros(STATE_DIMS)); self.register_buffer("state_sd", torch.ones(STATE_DIMS))
        self.register_buffer("act_mu", torch.zeros(horizon, ACT_DIMS)); self.register_buffer("act_sd", torch.ones(horizon, ACT_DIMS))

    def features(self, state, x0, prefix_pool=None):
        s = (state[:, :STATE_DIMS].float() - self.state_mu) / self.state_sd
        a = (x0[:, : self.horizon, :ACT_DIMS].float() - self.act_mu) / self.act_sd
        parts = [s, a.flatten(1)]
        if self.use_prefix:
            parts.append(self.prefix_proj(prefix_pool.float()))
        return torch.cat(parts, dim=1)

    def forward(self, ctx: dict, x0: torch.Tensor) -> torch.Tensor:
        pp = None
        if self.use_prefix:
            pp = ctx.get("prefix_pool")
            if pp is None:
                pp = pool_prefix(ctx["prefix_embs"], ctx["prefix_pad_masks"])
        return self.mlp(self.features(ctx["state"], x0, pp)).squeeze(-1)


def save_q(model: QNet, path: str | Path, meta: dict) -> None:
    path = Path(path)
    torch.save({"state_dict": model.state_dict(), "arch": model.arch, "meta": meta}, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))


def load_q(path: str | Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = QNet(**ck["arch"])
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m
