"""Do independently trained Qs agree on the action gradient, and does a step along one Q's
gradient raise the others? Evaluated on the half of episodes none of the judged Qs trained on."""
import sys, glob, math, numpy as np, torch
sys.path.insert(0, "scripts/phase2")
from q_model import load_q
S = sys.argv[1]
Q = {h: [load_q(f"{S}/q/{h}_s{s}.pt", "cpu") for s in range(5)] for h in ("hi", "lo")}
files = sorted(glob.glob("outputs/phase2/dumps/base50/task*_ep*.npz"))
def grad(q, ctx, x):
    x = x.clone().requires_grad_(True); out = q(ctx, x)
    g = torch.autograd.grad(out.sum(), x)[0][:, :, :7]
    return g, out.detach()
def unit(g): return g / g.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)
cos_pairs, cos_ens, raw_norm = [], [], []
res = {rho: {"self": [], "other": [], "ens_to_heldout": [], "rand": []} for rho in (0.002, 0.005, 0.01, 0.02)}
tasks = []
g_rng = torch.Generator().manual_seed(0)
for f in files[::3]:
    z = np.load(f); ep = int(z["episode_idx"]); qs = Q["hi"] if ep < 25 else Q["lo"]   # Qs that never saw this episode
    ctx = {"state": torch.tensor(z["state"].astype(np.float32)), "prefix_pool": torch.tensor(z["prefix_pool"].astype(np.float32))}
    x = torch.tensor(z["actions"].astype(np.float32))
    gs, outs = zip(*[grad(q, ctx, x) for q in qs])
    U = [unit(g) for g in gs]
    raw_norm.append(torch.stack([g.flatten(1).norm(dim=1) for g in gs]).mean().item())
    for i in range(5):
        for j in range(i + 1, 5):
            cos_pairs.append((U[i] * U[j]).flatten(1).sum(1).numpy())
    for rho in res:
        scale = rho * math.sqrt(320)   # same norm the sampler uses (numel = 10*32)
        def step(u):
            d = torch.zeros_like(x); d[:, :, :7] = u * scale; return x + d
        with torch.no_grad():
            # single Q 0 gradient: gain under itself vs under the 4 other seeds
            x1 = step(U[0])
            res[rho]["self"].append((qs[0](ctx, x1) - outs[0]).numpy())
            res[rho]["other"].append(torch.stack([qs[k](ctx, x1) - outs[k] for k in range(1, 5)]).mean(0).numpy())
            # ensemble of seeds 0-3 gradient: gain under held-out seed 4
            ue = unit(torch.stack(gs[:4]).mean(0)); x2 = step(ue)
            res[rho]["ens_to_heldout"].append((qs[4](ctx, x2) - outs[4]).numpy())
            ur = unit(torch.randn(x[:, :, :7].shape, generator=g_rng)); x3 = step(ur)
            res[rho]["rand"].append(torch.stack([qs[k](ctx, x3) - outs[k] for k in range(5)]).mean(0).numpy())
c = np.concatenate
print("mean raw gradient norm (logit per unit action, 70 dims): %.1f" % np.mean(raw_norm))
print("pairwise gradient cosine between seeds (same training half): mean %.3f, median %.3f, frac>0 %.2f" % (c(cos_pairs).mean(), np.median(c(cos_pairs)), (c(cos_pairs) > 0).mean()))
print(f"{'rho':>6} {'step_rms7':>9} | dlogit: {'self':>7} {'other seeds':>11} {'ens(4)->held-out seed':>22} {'random dir':>10}")
for rho, r in res.items():
    print(f"{rho:>6} {rho*math.sqrt(320/70):>9.4f} |         {c(r['self']).mean():>7.2f} {c(r['other']).mean():>11.2f} {c(r['ens_to_heldout']).mean():>22.2f} {c(r['rand']).mean():>10.2f}")
