"""CPU test: synthetic dumps -> train_q -> load_q -> guidance through GuidedSampler on the fake pi0.

Run: python3 scripts/phase2/test_q_model.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_guided_sampler as tgs  # installs the openpi stub, provides FakePi0  # noqa: E402
from guided_sampler import GuidanceConfig, GuidedSampler  # noqa: E402
from q_model import load_q  # noqa: E402
import train_q  # noqa: E402

H, A, S, D, STEPS = 10, 32, 32, 16, 5


def make_dumps(d: Path, n_tasks=3, n_eps=16, seed=0):
    rng = np.random.RandomState(seed)
    u_s = rng.randn(8); u_a = rng.randn(H, 7); u_p = rng.randn(D)
    for tid in range(n_tasks):
        for ep in range(n_eps):
            K = rng.randint(20, 40)
            # hidden episode quality z leaves a trace in every inference's state, actions and context,
            # the way a good or bad grasp would; label = z > 0 (+ noise)
            z = rng.randn()
            state = (rng.randn(K, S) * 0.5).astype(np.float32); state[:, :8] += 0.5 * z * u_s
            pool = (rng.randn(K, D) * 0.5).astype(np.float32); pool += 0.5 * z * u_p
            x0h = rng.randn(K, STEPS, H, A).astype(np.float32); x0h[:, :, :, :7] += 0.5 * z * u_a
            act = x0h[:, -1] + 0.1 * rng.randn(K, H, A).astype(np.float32)
            score = z + 0.3 * rng.randn()
            np.savez_compressed(d / f"task{tid:02d}_ep{ep:03d}.npz", task_id=tid, task=f"task {tid}", episode_idx=ep,
                                success=bool(score > 0), n_env_steps=K * 5, step_t=np.arange(K) * 5,
                                noise=x0h[:, 0], state=state, prefix_pool=pool, x0_hats=x0h,
                                times=np.linspace(1, 0.2, STEPS, dtype=np.float32), actions=act,
                                exec_actions=act[:, :5, :7], q_values=np.zeros(0))


def main():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "dumps"; d.mkdir()
        make_dumps(d)
        out = Path(td) / "q" / "q.pt"
        args = SimpleNamespace(dumps=[str(d)], out=str(out), train_on="both", val_frac=0.25, epochs=30, bs=256, lr=2e-3,
                               wd=1e-2, hidden=64, dropout=0.0, late_weight=0.0, no_prefix=False, seed=0)
        meta = train_q.main(args)
        assert meta["val_auc"] > 0.75, f"val AUC too low on a learnable synthetic rule: {meta['val_auc']}"
        assert out.exists() and out.with_suffix(".json").exists()
        print(f"ok: trainer learns the synthetic rule (val AUC {meta['val_auc']:.3f}, episode AUC {meta['val_episode_auc']:.3f})")

        q = load_q(out, torch.device("cpu"))
        g = torch.Generator().manual_seed(3)
        model = tgs.FakePi0()
        obs = tgs._obs(2, g); noise = torch.randn(2, H, A, generator=g)
        base = GuidedSampler(model)(torch.device("cpu"), obs, noise=noise, num_steps=5)
        hook = GuidedSampler(model, q_fn=q, cfg=GuidanceConfig(rho=0.05), record=True)
        out_g = hook(torch.device("cpu"), obs, noise=noise, num_steps=5)
        assert not torch.equal(base, out_g), "learned Q guidance must change the sample"
        with torch.no_grad():
            ctx = {"state": obs.state, "prefix_pool": torch.randn(2, D)}
            q_base, q_g = q(ctx, base), q(ctx, out_g)
        print(f"ok: loaded Q drives GuidedSampler through prefix_embs pooling; Q(base)={q_base.mean():.3f} Q(guided)={q_g.mean():.3f}")
        assert len(hook.records[-1].q_values) == 5
        # rho=0 with the learned Q attached is still the base sampler
        same = GuidedSampler(model, q_fn=q, cfg=GuidanceConfig(rho=0.0))(torch.device("cpu"), obs, noise=noise, num_steps=5)
        assert torch.equal(same, base)
        print("ok: rho=0 with learned Q attached is bitwise the base sampler")
        print("ALL PASS")


if __name__ == "__main__":
    main()
