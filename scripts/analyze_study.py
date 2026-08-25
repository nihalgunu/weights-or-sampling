#!/usr/bin/env python3
"""
Aggregate the TFG-vs-GRPO study matrices into the paper/email tables.

Reads outputs/study_{clip,clip_ocr,pickscore}_matrix.json (whichever exist) and
prints, per reward:
  1. Headline: best TFG arm vs best RL arm vs stacked, paired Δ vs no_lora on
     the PRIMARY scorer (the reward both methods optimize), 3 seeds x 30 prompts.
  2. TFG dose-response in rho.
  3. Amortization decomposition: RL Δ / TFG Δ; stacked vs TFG-alone
     (complementarity: does the banked part overlap TFG's gain or add to it?).
  4. Cross-metric hacking probe: each arm's Δ on the two non-primary scorers
     (+ jpeg — a jpeg spike alongside a primary win = de-texturing/blur hack).

Works on partial JSONs (missing summary block / missing arms): recomputes all
stats from raw scores.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
REWARD_KEYS = ["clip", "clip_ocr", "pickscore"]


def paired(sc_arm, sc_base, metric, seed_bases):
    deltas = []
    for b in seed_bases:
        b = str(b)
        if b not in sc_arm.get(metric, {}) or b not in sc_base.get(metric, {}):
            continue
        deltas += [a - c for a, c in zip(sc_arm[metric][b], sc_base[metric][b])]
    if not deltas:
        return None
    dm = statistics.mean(deltas)
    dsd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    se = dsd / len(deltas) ** 0.5 if deltas else 0.0
    return {
        "d": dm, "se": se, "z": dm / se if se > 0 else 0.0,
        "wins": sum(1 for x in deltas if x > 0), "n": len(deltas),
    }


def fmt(e):
    if e is None:
        return "  (missing)"
    return f"d={e['d']:+.4f} se={e['se']:.4f} z={e['z']:+.1f} wins={e['wins']}/{e['n']}"


def analyze(key: str):
    path = ROOT / "outputs" / f"study_{key}_matrix.json"
    if not path.exists():
        print(f"\n##### {key}: {path.name} missing — skipped")
        return
    d = json.loads(path.read_text())
    meta = d["meta"]
    prim = meta["reward"]
    seed_bases = meta["seed_bases"]
    scores = d["scores"]
    if "no_lora" not in scores:
        print(f"\n##### {key}: no no_lora arm yet — skipped")
        return
    base = scores["no_lora"]

    print(f"\n{'='*74}\n##### {key}  (primary scorer: {prim}; "
          f"{len(seed_bases)} seed bases x {meta['num_prompts']} prompts)")

    tfg_arms = sorted([a for a in scores if a.startswith("tfg_rho")],
                      key=lambda a: float(a[7:]))
    rl_arms = [a for a in scores if a.startswith("checkpoint")]
    stack_arms = [a for a in scores if a.startswith("stack_")]

    print("\n-- primary-scorer paired Δ vs no_lora --")
    for a in tfg_arms + rl_arms + stack_arms:
        print(f"  {a:34s} {fmt(paired(scores[a], base, prim, seed_bases))}")

    # Amortization decomposition
    best_tfg = max(tfg_arms, key=lambda a: paired(scores[a], base, prim, seed_bases)["d"],
                   default=None)
    best_rl = max(rl_arms, key=lambda a: paired(scores[a], base, prim, seed_bases)["d"],
                  default=None)
    if best_tfg and best_rl:
        dt = paired(scores[best_tfg], base, prim, seed_bases)["d"]
        dr = paired(scores[best_rl], base, prim, seed_bases)["d"]
        print(f"\n-- decomposition --  TFG*={best_tfg} Δ={dt:+.4f}   RL*={best_rl} Δ={dr:+.4f}")
        if dt != 0:
            print(f"   RL banks {dr/dt*100:.0f}% of TFG's gain; residual {100-dr/dt*100:.0f}%")
        for a in stack_arms:
            ds = paired(scores[a], base, prim, seed_bases)["d"]
            print(f"   {a}: Δ={ds:+.4f}  vs TFG-alone {dt:+.4f} "
                  f"({'stacks above' if ds > dt else 'below'} TFG-alone; "
                  f"vs RL-alone {dr:+.4f})")
            # paired stacked-vs-TFG-alone test
            e = paired(scores[a], scores[best_tfg], prim, seed_bases)
            print(f"     stacked-minus-TFG paired: {fmt(e)}")

    print("\n-- cross-metric probe (Δ vs no_lora on non-primary scorers) --")
    others = [mkey for mkey in ["clip", "pickscore", "jpeg"] if mkey != prim]
    for a in tfg_arms + rl_arms + stack_arms:
        cells = "  ".join(f"{mkey}:{fmt(paired(scores[a], base, mkey, seed_bases))}"
                          for mkey in others)
        print(f"  {a:34s} {cells}")


if __name__ == "__main__":
    keys = sys.argv[1:] or REWARD_KEYS
    for k in keys:
        analyze(k)
