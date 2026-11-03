"""Train the success-value model Q on dumps from libero_eval_guided.py.

Each dump (task{tt}_ep{eee}.npz) holds every inference of one episode plus the episode's
success label. Episodes are split into train/val by episode (never by inference), stratified
by task. Reports held-out AUC overall and per task, and saves q.pt + q.json.

Usage: python scripts/phase2/train_q.py --dumps outputs/phase2/dumps/m1 --out outputs/phase2/q/q.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q_model import ACT_DIMS, STATE_DIMS, QNet, save_q  # noqa: E402


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    n1, n0 = labels.sum(), (1 - labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    for v in np.unique(scores):
        idx = np.where(scores == v)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_dumps(dirs, train_on):
    files = sorted(f for d in dirs for f in glob.glob(str(Path(d) / "task*_ep*.npz")))
    assert files, f"no dumps under {dirs}"
    eps = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        K = z["actions"].shape[0]
        chunks = []
        if train_on in ("final", "both"):
            chunks.append(z["actions"][:, :, :ACT_DIMS].astype(np.float32))
        if train_on in ("x0hats", "both"):
            xh = z["x0_hats"].astype(np.float32)  # [K, steps, H, A]
            for s in range(xh.shape[1]):
                chunks.append(xh[:, s, :, :ACT_DIMS])
        n_rep = len(chunks)
        eps.append({
            "file": f, "task_id": int(z["task_id"]), "episode_idx": int(z["episode_idx"]), "success": bool(z["success"]),
            "state": np.tile(z["state"][:, :STATE_DIMS].astype(np.float32), (n_rep, 1)),
            "prefix_pool": np.tile(z["prefix_pool"].astype(np.float32), (n_rep, 1)),
            "x0": np.concatenate(chunks, 0),
            "frac": np.tile(np.arange(K, dtype=np.float32) / max(K - 1, 1), n_rep),
        })
    return eps


def split_episodes(eps, val_frac, seed):
    rng = np.random.RandomState(seed)
    train, val = [], []
    for tid in sorted({e["task_id"] for e in eps}):
        for lab in (True, False):
            group = [e for e in eps if e["task_id"] == tid and e["success"] == lab]
            rng.shuffle(group)
            n_val = int(round(len(group) * val_frac))
            val += group[:n_val]; train += group[n_val:]
    return train, val


def stack(eps):
    out = {k: np.concatenate([e[k] for e in eps], 0) for k in ("state", "prefix_pool", "x0", "frac")}
    out["y"] = np.concatenate([np.full(len(e["x0"]), float(e["success"]), np.float32) for e in eps])
    out["task"] = np.concatenate([np.full(len(e["x0"]), e["task_id"]) for e in eps])
    return out


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = load_dumps(a.dumps, a.train_on)
    n_succ = sum(e["success"] for e in eps)
    print(f"{len(eps)} episodes ({n_succ} success), {sum(len(e['x0']) for e in eps)} samples, tasks {sorted({e['task_id'] for e in eps})}")
    tr_eps, va_eps = split_episodes(eps, a.val_frac, a.seed)
    tr, va = stack(tr_eps), stack(va_eps)
    print(f"train {len(tr_eps)} eps / {len(tr['y'])} samples; val {len(va_eps)} eps / {len(va['y'])} samples")

    H = tr["x0"].shape[1]
    m = QNet(tr["prefix_pool"].shape[1], H, hidden=a.hidden, dropout=a.dropout, use_prefix=not a.no_prefix).to(dev)
    with torch.no_grad():
        m.state_mu.copy_(torch.tensor(tr["state"].mean(0))); m.state_sd.copy_(torch.tensor(tr["state"].std(0) + 1e-3))
        m.act_mu.copy_(torch.tensor(tr["x0"].mean(0))); m.act_sd.copy_(torch.tensor(tr["x0"].std(0) + 1e-3))
    T = {k: torch.tensor(v).to(dev) for k, v in tr.items() if k != "task"}
    V = {k: torch.tensor(v).to(dev) for k, v in va.items() if k != "task"}
    pos = float(tr["y"].mean()); pos_weight = torch.tensor((1 - pos) / max(pos, 1e-3), device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    n = len(T["y"]); best = (-1.0, None, -1)
    t0 = time.time()
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            logit = m({"state": T["state"][idx], "prefix_pool": T["prefix_pool"][idx]}, T["x0"][idx])
            w = 1.0 + a.late_weight * T["frac"][idx]  # optionally trust late-episode chunks more
            loss = (F.binary_cross_entropy_with_logits(logit, T["y"][idx], pos_weight=pos_weight, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        m.eval()
        with torch.no_grad():
            vs = m({"state": V["state"], "prefix_pool": V["prefix_pool"]}, V["x0"]).cpu().numpy() if len(va["y"]) else np.zeros(0)
        va_auc = auc(vs, va["y"]) if len(va["y"]) else float("nan")
        if (ep + 1) % max(1, a.epochs // 10) == 0 or ep == a.epochs - 1:
            print(f"ep {ep+1:3d} loss {tot/n:.4f} val_auc {va_auc:.3f} ({time.time()-t0:.0f}s)")
        if not np.isnan(va_auc) and va_auc > best[0]:
            best = (va_auc, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, ep + 1)
    if best[1] is not None:
        m.load_state_dict(best[1])
    m.eval()
    with torch.no_grad():
        vs = m({"state": V["state"], "prefix_pool": V["prefix_pool"]}, V["x0"]).cpu().numpy() if len(va["y"]) else np.zeros(0)
    per_task = {int(t): auc(vs[va["task"] == t], va["y"][va["task"] == t]) for t in np.unique(va["task"])} if len(va["y"]) else {}
    # episode-level: mean logit over the episode's inferences vs label (what BoN-by-Q relies on)
    ep_scores, ep_labels = [], []
    off = 0
    for e in va_eps:
        k = len(e["x0"]); ep_scores.append(float(vs[off:off + k].mean())); ep_labels.append(int(e["success"])); off += k
    meta = {"dumps": a.dumps, "train_on": a.train_on, "n_episodes": len(eps), "n_success": int(n_succ),
            "train_eps": len(tr_eps), "val_eps": len(va_eps), "val_auc": best[0], "best_epoch": best[2],
            "val_auc_per_task": per_task, "val_episode_auc": auc(ep_scores, ep_labels) if ep_labels else float("nan"),
            "use_prefix": not a.no_prefix, "hidden": a.hidden, "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
            "feature_dims": {"state": STATE_DIMS, "prefix": int(tr["prefix_pool"].shape[1]), "horizon": int(H), "act": ACT_DIMS}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_q(m.cpu(), a.out, meta)
    print("saved", a.out, json.dumps({k: meta[k] for k in ("val_auc", "best_epoch", "val_episode_auc", "val_auc_per_task")}))
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dumps", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train_on", choices=["final", "x0hats", "both"], default="both")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--late_weight", type=float, default=0.0)
    p.add_argument("--no_prefix", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
