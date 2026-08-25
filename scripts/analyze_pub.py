#!/usr/bin/env python3
"""
Publication-grade study analysis. Reads outputs/study_{key}_pub_matrix.json
(+ optional outputs/study_{key}_pub_external.json) and reports, per reward:

  1. Prompt-clustered stats (mean over eval seeds per prompt, t over prompts):
     each arm vs base; TFG vs each RL seed; stacked vs TFG.
  2. Across-training-seed variability of the RL and stacked arms
     (mean +/- range over the 3 training seeds).
  3. Pooled RL arm (all 3 training seeds) vs TFG.
  4. Additivity residual per training seed with CI.
  5. External metrics (SigLIP / OCR-match), prompt-clustered.
  6. Original-30 vs held-out-120 split for the DDRL-style prompt sets
     (RL trained on the first 30; the 120 are distribution-matched held-out).

All tests are conservative: clustered at the prompt level.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
KEYS = ["clip", "clip_ocr", "pickscore"]


def per_prompt(scores, arm_a, arm_b, metric, seed_bases, idx):
    out = []
    for i in idx:
        ds = []
        for b in seed_bases:
            b = str(b)
            va, vb = scores[arm_a][metric][b][i], scores[arm_b][metric][b][i]
            if va is None or vb is None:
                continue
            ds.append(va - vb)
        if ds:
            out.append(sum(ds) / len(ds))
    return out


def t_stat(deltas):
    n = len(deltas)
    if n < 2:
        return None
    m = statistics.mean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(n)
    return {"d": m, "se": se, "t": m / se if se > 0 else 0.0,
            "wins": sum(1 for x in deltas if x > 0), "n": n}


def fmt(e):
    if e is None:
        return "(insufficient)"
    return f"d={e['d']:+.4f} se={e['se']:.4f} t={e['t']:+.1f} wins={e['wins']}/{e['n']}"


def analyze(key):
    mpath = ROOT / "outputs" / f"study_{key}_pub_matrix.json"
    if not mpath.exists():
        print(f"\n##### {key}: missing {mpath.name}")
        return
    d = json.loads(mpath.read_text())
    meta, sc = d["meta"], d["scores"]
    prim, sb, n = meta["reward"], meta["seed_bases"], meta["num_prompts"]
    all_idx = list(range(n))
    print(f"\n{'='*76}\n##### {key}  (primary {prim}; {n} prompts x {len(sb)} eval seeds, clustered n={n})")

    tfg_arms = sorted([a for a in sc if a.startswith("tfg_rho")], key=lambda a: float(a[7:]))
    rl_arms = sorted([a for a in sc if a.startswith("rl_s")])
    stack_arms = sorted([a for a in sc if a.startswith("stack_")])

    print("\n-- arm vs base (prompt-clustered) --")
    for a in tfg_arms + rl_arms + stack_arms:
        print(f"  {a:26s} {fmt(t_stat(per_prompt(sc, a, 'no_lora', prim, sb, all_idx)))}")

    # Across-training-seed variability
    if rl_arms:
        rl_ds = [t_stat(per_prompt(sc, a, "no_lora", prim, sb, all_idx))["d"] for a in rl_arms]
        print(f"\n-- RL across training seeds: mean {statistics.mean(rl_ds):+.4f}  "
              f"range [{min(rl_ds):+.4f}, {max(rl_ds):+.4f}]  "
              f"sd {statistics.stdev(rl_ds):.4f}" if len(rl_ds) > 1 else "")

    # TFG headline vs pooled RL (per prompt: TFG delta minus mean-over-seeds RL delta)
    if tfg_arms and rl_arms:
        best_tfg = max(tfg_arms, key=lambda a: t_stat(per_prompt(sc, a, "no_lora", prim, sb, all_idx))["d"])
        pooled = []
        for i in all_idx:
            tfg_d, rl_d = [], []
            for b in sb:
                b = str(b)
                base_v = sc["no_lora"][prim][b][i]
                tfg_d.append(sc[best_tfg][prim][b][i] - base_v)
                rl_d += [sc[a][prim][b][i] - base_v for a in rl_arms]
            pooled.append(sum(tfg_d) / len(tfg_d) - sum(rl_d) / len(rl_d))
        print(f"\n-- {best_tfg} minus pooled-RL(3 seeds): {fmt(t_stat(pooled))}")
        for a in rl_arms:
            e = t_stat(per_prompt(sc, best_tfg, a, prim, sb, all_idx))
            print(f"   vs {a:22s} {fmt(e)}")

    # Stacked vs TFG-alone (matched rho), per training seed
    for st in stack_arms:
        rho = st.rsplit("rho", 1)[1]
        tfg_ref = f"tfg_rho{rho}"
        if tfg_ref in sc:
            e = t_stat(per_prompt(sc, st, tfg_ref, prim, sb, all_idx))
            print(f"-- {st} minus {tfg_ref}: {fmt(e)}")

    # Additivity residual per training seed (matched-rho stacks only)
    print("\n-- additivity residual stack - (RL + TFG), per training seed --")
    for st in stack_arms:
        seed = st.split("_")[1]
        rho = st.rsplit("rho", 1)[1]
        rl_ref, tfg_ref = f"rl_{seed}", f"tfg_rho{rho}"
        if rl_ref in sc and tfg_ref in sc:
            resid = []
            for i in all_idx:
                vals = []
                for b in sb:
                    b = str(b)
                    base_v = sc["no_lora"][prim][b][i]
                    vals.append((sc[st][prim][b][i] - base_v)
                                - (sc[rl_ref][prim][b][i] - base_v)
                                - (sc[tfg_ref][prim][b][i] - base_v))
                resid.append(sum(vals) / len(vals))
            e = t_stat(resid)
            print(f"  {st:26s} {fmt(e)}  95% CI [{e['d']-2*e['se']:+.4f}, {e['d']+2*e['se']:+.4f}]")

    # Train-30 vs held-out-120 split (DDRL-style sets only)
    if key in ("clip", "pickscore") and n > 30:
        print("\n-- split: prompts 0-29 (RL-trained) vs 30+ (held-out, same distribution) --")
        for a in (tfg_arms[-1:] + rl_arms + stack_arms[:1]):
            e_tr = t_stat(per_prompt(sc, a, "no_lora", prim, sb, list(range(30))))
            e_ho = t_stat(per_prompt(sc, a, "no_lora", prim, sb, list(range(30, n))))
            print(f"  {a:26s} train30: {fmt(e_tr)}")
            print(f"  {'':26s} held120: {fmt(e_ho)}")

    # External metrics
    epath = ROOT / "outputs" / f"study_{key}_pub_external.json"
    if epath.exists():
        ed = json.loads(epath.read_text())
        esc = ed.get("scores", {})
        if "no_lora" in esc:
            print("\n-- external metrics (prompt-clustered vs base) --")
            for m in ed["meta"]["metrics"]:
                for a in tfg_arms + rl_arms + stack_arms:
                    if a in esc and m in esc[a] and esc[a][m]:
                        e = t_stat(per_prompt(esc, a, "no_lora", m, list(esc[a][m].keys()), all_idx))
                        print(f"  {m:7s} {a:26s} {fmt(e)}")


if __name__ == "__main__":
    for k in (sys.argv[1:] or KEYS):
        analyze(k)
