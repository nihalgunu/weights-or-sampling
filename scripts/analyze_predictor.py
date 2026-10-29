#!/usr/bin/env python3
"""
Two-factor pre-flight predictor over all reward settings.

Per setting:
  slope      least-squares-through-origin slope of the TFG dose-response
             delta(rho) over rho in {2,5,10} (coordinate 1; "steepness"); rho=20 is
             held out so the slope->gain check is not circular
  slope_n    slope divided by the SD of base scores across prompts (scale-free)
  cos        cross-prompt gradient cosine from study_pdep*.json (coordinate 2)
  d_tfg20    TFG rho=20 vs base (150-prompt matrix when available)
  d_rl       best-regime RL vs base, mean over seeds (beta=0 where available)
  share      d_rl / d_tfg20   (how much of the guidance gain RL amortises)
  ratio      d_tfg20 / d_rl
Then leave-one-out: predict `share` from (slope_n, cos) with a 2-feature linear
fit; report Spearman correlations and LOO error. 5 settings exist today; the
rw8 run adds hpsv2, siglip, clip_pickapic.

Usage: python3 scripts/analyze_predictor.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyze_pub import per_prompt, t_stat  # noqa: E402

OUT = ROOT / "outputs"


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def primary_scores(d, metric):
    """Return scores dict with an 'ensemble' metric synthesised if needed."""
    sc = d["scores"]
    if metric == "ensemble":
        for arm in sc:
            if "ensemble" not in sc[arm]:
                sc[arm]["ensemble"] = {b: [None if (c is None or p is None) else 0.5 * c + 0.5 * p
                                          for c, p in zip(sc[arm]["clip"][b], sc[arm]["pickscore"][b])]
                                      for b in sc[arm]["clip"]}
    return sc


def delta(d, arm, metric):
    if d is None or arm not in d["scores"]:
        return None
    sc = primary_scores(d, metric)
    meta = d["meta"]
    e = t_stat(per_prompt(sc, arm, "no_lora", metric, meta["seed_bases"], list(range(meta["num_prompts"]))))
    return e["d"] if e else None


def base_sd(d, metric):
    sc = primary_scores(d, metric)
    vals = []
    for b in d["meta"]["seed_bases"]:
        vals += [v for v in sc["no_lora"][metric][str(b)] if v is not None]
    return statistics.pstdev(vals) if len(vals) > 1 else None


def slope_through_origin(points):
    pts = [(r, dv) for r, dv in points if dv is not None]
    if len(pts) < 2:
        return None, len(pts)
    num = sum(r * dv for r, dv in pts)
    den = sum(r * r for r, _ in pts)
    return num / den, len(pts)


def dose(d, metric, rhos=(2, 5, 10, 20)):
    return [(r, delta(d, f"tfg_rho{r:g}", metric)) for r in rhos]


def rl_mean(pairs):
    vals = [delta(d, a, m) for d, a, m in pairs]
    vals = [v for v in vals if v is not None]
    return (statistics.mean(vals), len(vals)) if vals else (None, 0)


def settings():
    pdep = load("study_pdep.json") or {}
    pdep8 = load("study_pdep_rw8.json") or {}
    S = []
    # --- existing five ---
    dC, pC, bC = load("study_clip_matrix.json"), load("study_clip_pub_matrix.json"), load("study_beta0_clip_matrix.json")
    S.append(dict(name="clip_geneval", metric="clip", dose=dose(dC, "clip"), sd=base_sd(pC or dC, "clip"),
                  tfg20=delta(pC, "tfg_rho20", "clip"),
                  rl=rl_mean([(bC, "rl_b0_s42", "clip"), (bC, "rl_b0_s1042", "clip")]),
                  rl_note="beta0 x2", cos=pdep.get("clip_geneval", {}).get("mean_pairwise_cos")))
    dO, pO, bO = load("study_clip_ocr_matrix.json"), load("study_clip_ocr_pub_matrix.json"), load("study_beta0_ocr_matrix.json")
    S.append(dict(name="clip_ocr", metric="clip", dose=dose(dO, "clip"), sd=base_sd(pO or dO, "clip"),
                  tfg20=delta(pO, "tfg_rho20", "clip"),
                  rl=rl_mean([(bO, "rl_ocr_b0", "clip")]), rl_note="beta0 x1",
                  cos=pdep.get("clip_ocr", {}).get("mean_pairwise_cos")))
    dP, pP, kP = load("study_pickscore_matrix.json"), load("study_pickscore_pub_matrix.json"), load("study_kl_matrix.json")
    # RL at the tuned regime (beta=0, from the KL sweep), consistent with the other settings.
    S.append(dict(name="pickscore_geneval", metric="pickscore", dose=dose(dP, "pickscore"), sd=base_sd(pP or dP, "pickscore"),
                  tfg20=delta(pP, "tfg_rho20", "pickscore"),
                  rl=rl_mean([(kP, "rl_beta0", "pickscore")]) if kP else rl_mean([(pP, f"rl_s{s}", "pickscore") for s in (42, 1042, 2042)]),
                  rl_note="beta0 x1 (kl sweep)" if kP else "beta0.04 x3",
                  cos=pdep.get("pickscore_geneval", {}).get("mean_pairwise_cos")))
    dA, aS = load("study_rw6_aesthetic_matrix.json"), load("study_aes_seeds_matrix.json")
    S.append(dict(name="aesthetic", metric="aesthetic", dose=dose(dA, "aesthetic"), sd=base_sd(dA, "aesthetic") if dA else None,
                  tfg20=delta(dA, "tfg_rho20", "aesthetic"),
                  rl=rl_mean([(dA, "rl_b0_s42", "aesthetic"), (aS, "rl_b0_s1042", "aesthetic"), (aS, "rl_b0_s2042", "aesthetic")]),
                  rl_note="beta0 x3", cos=pdep.get("aesthetic", {}).get("mean_pairwise_cos")))
    dE, bE = load("study_ensemble_matrix.json"), load("study_beta0_ens_matrix.json")
    dEd = load("study_rw8_ensemble_dose_matrix.json") or load("checkpoints/study_rw8_ensemble_dose_matrix.json")
    ens_dose = ([(r, delta(dEd, f"tfg_rho{r}", "ensemble")) for r in (2, 5, 10)] if dEd else []) + \
               ([(20, delta(dE, "tfg_rho20", "ensemble"))] if dE else [])
    S.append(dict(name="ensemble_geneval", metric="ensemble", dose=ens_dose,
                  sd=base_sd(dE, "ensemble") if dE else None, tfg20=delta(dE, "tfg_rho20", "ensemble"),
                  rl=rl_mean([(bE, "rl_ens_b0", "ensemble")]), rl_note="beta0 x1" + ("" if dEd else " (slope from rho20 only)"),
                  cos=pdep.get("ensemble_geneval", {}).get("mean_pairwise_cos")))
    # --- rw8 additions ---
    for key, metric, cos_key in (("hpsv2", "hpsv2", "hpsv2_geneval"), ("siglip", "siglip", "siglip_geneval"),
                                 ("clip_pickapic", "clip", "clip_pickapic")):
        d8 = load(f"study_rw8_{key}_matrix.json")
        d8t = d8 or load(f"study_rw8_{key}_tfg_matrix.json")   # TFG half (A100-able)
        d8r = d8 or load(f"study_rw8_{key}_rl_matrix.json")    # RL half (80GB)
        d8s = load(f"study_rw8_{key}_rl_s43_44_matrix.json")   # extra GRPO seeds (Oct 28)
        rl_pairs = ([(d8r, "rl_b0_s42", metric)] if d8r else []) + \
                   ([(d8s, "rl_b0_s43", metric), (d8s, "rl_b0_s44", metric)] if d8s else [])
        S.append(dict(name=cos_key, metric=metric, dose=dose(d8t, metric) if d8t else [], sd=base_sd(d8t, metric) if d8t else None,
                      tfg20=delta(d8t, "tfg_rho20", metric) if d8t else None,
                      rl=rl_mean(rl_pairs) if rl_pairs else (None, 0), rl_note=f"beta0 x{len(rl_pairs)}",
                      rl_seeds=[delta(d, a, m) for d, a, m in rl_pairs],
                      cos=pdep8.get(cos_key, {}).get("mean_pairwise_cos")))
    return S


def spearman(x, y):
    n = len(x)
    if n < 3:
        return None
    rx = [sorted(x).index(v) + 1 for v in x]
    ry = [sorted(y).index(v) + 1 for v in y]
    dsq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * dsq / (n * (n * n - 1))


def loo_linear(X, y):
    """Leave-one-out 2-feature linear regression (with intercept) via numpy."""
    import numpy as np
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = len(y)
    if n < 4:
        return None
    preds = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        A = np.c_[np.ones(m.sum()), X[m]]
        w, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        preds.append(float(np.r_[1.0, X[i]] @ w))
    preds = np.array(preds)
    ss_res = float(((y - preds) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"loo_r2": 1 - ss_res / ss_tot if ss_tot > 0 else None, "loo_mae": float(np.abs(y - preds).mean()),
            "pred": preds.tolist()}


def main():
    S = settings()
    rows = []
    # Gains are on each reward's own scale; divide by the base score SD across prompts
    # (tfg_n, rl_n) so settings are comparable. `share` (RL/TFG at rho20) is scale-free
    # but conflates the two coordinates and depends on rho20 being matched strength.
    print(f"{'setting':20s} {'slope_n':>8s} {'cos':>6s} {'d_tfg20':>8s} {'tfg_n':>6s} {'d_rl':>8s} {'rl_n':>6s} {'share':>6s}  rl")
    for s in S:
        # Fit the slope on rho <= 10 only, so the rho=20 gain it is compared with is
        # out of sample (fitting through rho=20 and "predicting" rho=20 is circular).
        sl, npts = slope_through_origin([(r, v) for r, v in s["dose"] if r <= 10])
        sl_n = (sl / s["sd"]) if (sl is not None and s["sd"]) else None
        rl, nrl = s["rl"]
        share = (rl / s["tfg20"]) if (rl is not None and s["tfg20"]) else None
        ratio = (s["tfg20"] / rl) if (rl not in (None, 0) and s["tfg20"] is not None) else None
        tfg_n = (s["tfg20"] / s["sd"]) if (s["tfg20"] is not None and s["sd"]) else None
        rl_n = (rl / s["sd"]) if (rl is not None and s["sd"]) else None
        f = lambda v, w=8, p=4: (f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'--':>{w}s}")
        seeds = s.get("rl_seeds") or []
        seed_txt = ("  seeds[" + ", ".join(f"{v:+.4f}" for v in seeds if v is not None) + "]"
                    + (f" sd={statistics.stdev([v for v in seeds if v is not None]):.4f}" if len([v for v in seeds if v is not None]) > 1 else "")) if seeds else ""
        print(f"{s['name']:20s} {f(sl_n, 8, 3)} {f(s['cos'], 6, 3)} {f(s['tfg20'])} {f(tfg_n, 6, 2)} {f(rl)} {f(rl_n, 6, 2)} {f(share, 6, 2)}  {s['rl_note']} (dose pts {npts}){seed_txt}")
        if None not in (sl_n, s["cos"], share):
            rows.append((s["name"], sl_n, s["cos"], share, math.log(ratio) if ratio and ratio > 0 else None, tfg_n, rl_n))
    if len(rows) >= 3:
        names = [r[0] for r in rows]
        print(f"\n{len(rows)} settings with all coordinates: {names}")
        print(f"  spearman(slope_n, tfg_n)  = {spearman([r[1] for r in rows], [r[5] for r in rows])}   (coordinate 1 -> guidance response)")
        print(f"  spearman(cos, rl_n)       = {spearman([r[2] for r in rows], [r[6] for r in rows])}   (coordinate 2 -> RL amortisation)")
        print(f"  spearman(slope_n, share)  = {spearman([r[1] for r in rows], [r[3] for r in rows])}")
        print(f"  spearman(cos, share)      = {spearman([r[2] for r in rows], [r[3] for r in rows])}")
        res = loo_linear([[r[1], r[2]] for r in rows], [r[3] for r in rows])
        if res:
            print(f"  LOO linear share ~ slope_n + cos: R2 = {res['loo_r2']:.3f}  MAE = {res['loo_mae']:.3f}")
            for r, p in zip(rows, res["pred"]):
                print(f"    {r[0]:20s} share {r[3]:+.3f}  predicted {p:+.3f}")
    else:
        print("\n(need >= 3 complete settings for correlations)")


if __name__ == "__main__":
    main()
