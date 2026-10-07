#!/usr/bin/env python3
"""
DRaFT arm analysis: each DRaFT seed vs base, prompt-clustered, next to the
existing TFG / GRPO numbers for the same reward setting (pub + beta=0 matrices),
so the fourth cell of the 2x2 reads off one table.

Usage: python3 scripts/analyze_draft.py [pilot|clip|clip_ocr|pickscore|aesthetic ...]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyze_pub import fmt, per_prompt, t_stat  # noqa: E402

REF = {  # reward key -> (matrix files with reference arms, primary metric)
    "clip": (["study_clip_pub_matrix.json", "study_beta0_clip_matrix.json"], "clip"),
    "clip_ocr": (["study_clip_ocr_pub_matrix.json", "study_beta0_ocr_matrix.json"], "clip"),
    "pickscore": (["study_pickscore_pub_matrix.json"], "pickscore"),
    "aesthetic": (["study_rw6_aesthetic_matrix.json", "study_aes_seeds_matrix.json"], "aesthetic"),
    "pilot": ([], "clip"),
}


def load(name):
    p = ROOT / "outputs" / name
    return json.loads(p.read_text()) if p.exists() else None


def arm_stats(d, arms, metric=None):
    meta, sc = d["meta"], d["scores"]
    metric = metric or meta["reward"]
    sb, n = meta["seed_bases"], meta["num_prompts"]
    idx = list(range(n))
    return {a: t_stat(per_prompt(sc, a, "no_lora", metric, sb, idx)) for a in arms if a in sc}


def main(keys):
    for key in keys:
        fname = "study_draft_pilot_matrix.json" if key == "pilot" else f"study_{key}_draft_matrix.json"
        d = load(fname)
        print(f"\n{'=' * 76}\n##### DRaFT[{key}]  ({fname})")
        if d is None:
            print("  (matrix not present yet)")
            continue
        metric = REF.get(key, ([], d["meta"]["reward"]))[1]
        meta = d["meta"]
        print(f"  primary {metric}; {meta['num_prompts']} prompts x {len(meta['seed_bases'])} eval seeds")
        draft_arms = [a for a in d["scores"] if a != "no_lora"]
        st = arm_stats(d, draft_arms, metric)
        print("\n-- DRaFT arms vs base (prompt-clustered) --")
        for a, e in st.items():
            print(f"  {a:34s} {fmt(e)}")
        seeds = [e["d"] for a, e in st.items() if a.startswith("draft_s") and e]
        if len(seeds) > 1:
            print(f"  mean over {len(seeds)} seeds: {statistics.mean(seeds):+.4f}  "
                  f"range [{min(seeds):+.4f}, {max(seeds):+.4f}]")
        # cross-metric hacking probes
        for m in ("jpeg", "pickscore", "clip"):
            if m == metric or m not in d["scores"]["no_lora"]:
                continue
            xs = arm_stats(d, [a for a in draft_arms if a.startswith("draft_s")], m)
            vals = [e["d"] for e in xs.values() if e]
            if vals:
                print(f"  cross-metric {m:9s}: mean delta {statistics.mean(vals):+.4f}")
        # reference arms from the existing matrices
        refs = REF.get(key, ([], None))[0]
        if refs:
            print("\n-- reference arms (same reward, existing matrices) --")
        for rf in refs:
            r = load(rf)
            if r is None:
                continue
            arms = [a for a in r["scores"] if a.startswith(("tfg_rho20", "tfg_rho80", "rl_", "stack_s42"))]
            rs = arm_stats(r, arms, metric)
            for a, e in rs.items():
                print(f"  {rf[:28]:28s} {a:22s} {fmt(e)}")
            rl = [e["d"] for a, e in rs.items() if a.startswith("rl_") and e]
            if len(rl) > 1:
                print(f"  {rf[:28]:28s} RL mean {statistics.mean(rl):+.4f} over {len(rl)} seeds")


if __name__ == "__main__":
    main(sys.argv[1:] or ["pilot", "clip", "clip_ocr", "pickscore", "aesthetic"])
