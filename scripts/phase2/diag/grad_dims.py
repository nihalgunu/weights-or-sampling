import sys, glob, math, numpy as np, torch
sys.path.insert(0, "scripts/phase2")
from q_model import load_q
S = sys.argv[1]
hi = [load_q(f"{S}/q/hi_s{s}.pt", "cpu") for s in range(5)]; lo = [load_q(f"{S}/q/lo_s{s}.pt", "cpu") for s in range(5)]
np.set_printoptions(precision=3, suppress=True, linewidth=160)
print("act_sd per dim (mean over horizon), hi_s0:", hi[0].act_sd.mean(0).numpy())
print("act_mu per dim (mean over horizon), hi_s0:", hi[0].act_mu.mean(0).numpy())
files = sorted(glob.glob("outputs/phase2/dumps/base50/task*_ep*.npz"))
def grad(q, ctx, x):
    x = x.clone().requires_grad_(True); out = q(ctx, x)
    return torch.autograd.grad(out.sum(), x)[0][:, :, :7]
def unit(g): return g / g.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)
cos_x, share_d, share_h, sign_d = [], [], [], []
per_task = {}
for f in files[::3]:
    z = np.load(f); tid = int(z["task_id"]); ep = int(z["episode_idx"])
    ctx = {"state": torch.tensor(z["state"].astype(np.float32)), "prefix_pool": torch.tensor(z["prefix_pool"].astype(np.float32))}
    x = torch.tensor(z["actions"].astype(np.float32))
    gh = unit(torch.stack([grad(q, ctx, x) for q in hi]).mean(0)); gl = unit(torch.stack([grad(q, ctx, x) for q in lo]).mean(0))
    cx = (gh * gl).flatten(1).sum(1).numpy(); cos_x.append(cx); per_task.setdefault(tid, []).append(cx)
    g = gh if ep < 25 else gl   # held-out ensemble gradient
    share_d.append((g ** 2).sum(1).numpy()); share_h.append((g ** 2).sum(2).numpy()); sign_d.append(g.mean(1).numpy())
c = np.concatenate
print("cosine between ensemble gradients of the two disjoint halves: mean %.3f median %.3f frac>0 %.2f" % (c(cos_x).mean(), np.median(c(cos_x)), (c(cos_x) > 0).mean()))
print("  per task:", {t: round(float(c(v).mean()), 2) for t, v in sorted(per_task.items())})
print("gradient energy share by action dim (x y z rx ry rz grip):", c(share_d).mean(0))
print("mean signed gradient by action dim:", c(sign_d).mean(0))
print("gradient energy share by horizon step 0..9:", c(share_h).mean(0))
A = c([np.load(f)["actions"][:, :, :7] for f in files[::3]])
print("action sd by dim over dataset:", A.reshape(-1, 7).std(0)); print("action mean by dim:", A.reshape(-1, 7).mean(0))
print("grip dim value histogram:", np.histogram(A[..., 6].ravel(), bins=[-3, -1.5, -0.5, 0.5, 1.5, 3])[0])
