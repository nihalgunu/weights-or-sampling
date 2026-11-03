"""Diagnostics for a trained Q on held-out episodes: does it rank *actions*, or only states?

Reproduces the trainer's episode split (same --seed / --val_frac), then reports per task:
  auc_all      held-out AUC on every inference (same as train_q's number)
  auc_early    AUC using only the first --early inferences of each episode (before failure is visible)
  auc_swapA    AUC after replacing each sample's action chunk with a random chunk from another
               held-out sample of the same task (actions destroyed, context kept)
  auc_swapS    AUC after replacing state+prefix with another sample's (context destroyed, actions kept)
  dQ_action    mean |Q(ctx, a') - Q(ctx, a)| over the swap above; dQ_ctx the same for context swaps
  grad_ratio   mean ||dQ/dx0|| / ||dQ/dstate||  (what guidance actually uses vs what it cannot change)
  pair_acc     within-task, same-step-bucket pairs (one success ep, one failure ep): fraction where
               Q(success ctx, success chunk) > Q(success ctx, failure chunk)  -- counterfactual ranking
               with the context held fixed; 0.5 = Q ignores the chunk.
Usage: python scripts/phase2/analyze_q.py --q outputs/phase2/q/both_s0/q.pt --dumps outputs/phase2/dumps/base50
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from q_model import load_q, ACT_DIMS, STATE_DIMS  # noqa
from train_q import auc, load_dumps, split_episodes  # noqa


def q_eval(m, state, pool, x0):
    with torch.no_grad():
        return m({"state": torch.tensor(state), "prefix_pool": torch.tensor(pool)}, torch.tensor(x0)).numpy()


def main(a):
    m = load_q(a.q, "cpu")
    meta = json.load(open(Path(a.q).with_suffix(".json")))
    if a.episodes:  # cross-fitting: evaluate on the whole half the Q never saw
        lo, hi = (int(v) for v in a.episodes.split(":"))
        va = load_dumps([a.dumps], "final", (lo, hi))
    else:
        eps = load_dumps([a.dumps], "final")  # one row per inference: the executed chunk
        _, va = split_episodes(eps, a.val_frac, meta.get("seed", 0))
    rng = np.random.RandomState(0)
    rows = []
    for tid in sorted({e["task_id"] for e in va}):
        E = [e for e in va if e["task_id"] == tid]
        ys = np.array([e["success"] for e in E])
        if ys.all() or (~ys).all():
            rows.append((tid, len(E), int(ys.sum()), *([float("nan")] * 8))); continue
        S = np.concatenate([e["state"] for e in E]); P = np.concatenate([e["prefix_pool"] for e in E])
        X = np.concatenate([e["x0"] for e in E]); Y = np.concatenate([np.full(len(e["x0"]), e["success"]) for e in E]).astype(int)
        step = np.concatenate([np.arange(len(e["x0"])) for e in E])
        q = q_eval(m, S, P, X)
        perm = rng.permutation(len(X))
        q_swapA = q_eval(m, S, P, X[perm]); q_swapS = q_eval(m, S[perm], P[perm], X)
        early = step < a.early
        # gradient ratio
        st = torch.tensor(S[:512], requires_grad=True); x0 = torch.tensor(X[:512], requires_grad=True)
        with torch.enable_grad():
            out = m({"state": st, "prefix_pool": torch.tensor(P[:512])}, x0).sum()
            gs, gx = torch.autograd.grad(out, [st, x0])
        gr = float((gx.flatten(1).norm(dim=1) / gs[:, :STATE_DIMS].norm(dim=1).clamp_min(1e-9)).mean())
        # counterfactual pairs: success ctx, same step bucket, chunk from a failure episode
        succ_i = np.where(Y == 1)[0]; fail_i = np.where(Y == 0)[0]
        wins = tot = 0
        for i in succ_i[rng.permutation(len(succ_i))[:400]]:
            cand = fail_i[np.abs(step[fail_i] - step[i]) <= 1]
            if len(cand) == 0: continue
            j = rng.choice(cand)
            qa = q_eval(m, S[i:i+1], P[i:i+1], X[i:i+1])[0]; qb = q_eval(m, S[i:i+1], P[i:i+1], X[j:j+1])[0]
            wins += qa > qb; tot += 1
        rows.append((tid, len(E), int(ys.sum()), auc(q, Y), auc(q[early], Y[early]), auc(q_swapA, Y), auc(q_swapS, Y),
                     float(np.abs(q_swapA - q).mean()), float(np.abs(q_swapS - q).mean()), gr, wins / max(tot, 1)))
    print(f"{'task':>4} {'nval':>4} {'succ':>4} {'auc_all':>7} {'auc_early':>9} {'auc_swapA':>9} {'auc_swapS':>9} {'dQ_act':>6} {'dQ_ctx':>6} {'grad_ratio':>10} {'pair_acc':>8}")
    for r in rows:
        print(f"{r[0]:>4} {r[1]:>4} {r[2]:>4} " + " ".join(f"{v:{w}.3f}" for v, w in zip(r[3:], (7, 9, 9, 9, 6, 6, 10, 8))))
    ok = [r for r in rows if r[3] == r[3]]
    mean = lambda k: float(np.mean([r[k] for r in ok]))
    print(f"mean over informative tasks: auc_all {mean(3):.3f} auc_early {mean(4):.3f} auc_swapA {mean(5):.3f} auc_swapS {mean(6):.3f} "
          f"dQ_act {mean(7):.3f} dQ_ctx {mean(8):.3f} grad_ratio {mean(9):.3f} pair_acc {mean(10):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--q", required=True); p.add_argument("--dumps", required=True)
    p.add_argument("--val_frac", type=float, default=0.2); p.add_argument("--early", type=int, default=3)
    p.add_argument("--episodes", default="", help="lo:hi held-out episode range (cross-fitting)")
    main(p.parse_args())
