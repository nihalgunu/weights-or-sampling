#!/usr/bin/env python3
"""Pick the saved checkpoint closest to the peak of the smoothed TRAINING reward.

DRaFT without a KL term can over-optimise and drift; the DRaFT paper early-stops.
Selection uses only the training prompts' rewards (train_log.jsonl), so the
held-out eval prompts stay untouched.

Usage: python3 scripts/cloud/pick_best_ckpt.py <run_dir> [--window 50]
Prints: <checkpoint dir>/lora  (and a summary on stderr)
"""
import argparse, json, re, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--window", type=int, default=50)
a = ap.parse_args()
run = Path(a.run_dir)
rows = [json.loads(l) for l in open(run / "train_log.jsonl")]
r = [x["reward"] for x in rows]
w = min(a.window, len(r))
sm = [(i + w, sum(r[i:i + w]) / w) for i in range(0, len(r) - w + 1)]  # (step at window end, mean)
ckpts = sorted(((int(re.search(r"\d+", p.name).group()), p) for p in run.glob("checkpoint-*") if (p / "lora").is_dir()))
if not ckpts:
    sys.exit("no checkpoints")
peak_step, peak = max(sm, key=lambda t: t[1]) if sm else (ckpts[-1][0], None)
best = min(ckpts, key=lambda c: abs(c[0] - peak_step))
final = ckpts[-1]
print(f"[pick] peak smoothed train reward {peak} at step ~{peak_step}; best ckpt {best[1].name}, final {final[1].name}; "
      f"final-window mean {sm[-1][1] if sm else None}", file=sys.stderr)
print(best[1] / "lora")
