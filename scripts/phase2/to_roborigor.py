"""Convert libero_eval_guided dumps (one .npz per episode) into RoboRigor EpisodeRecord JSONL,
so arms can be compared with `roborigor report` (paired exact statistics, informative-task split).

Pairing: records from different arms share (task_id, init_id=episode_idx, seed, sampling_seed) when
they were run with the same --seed, so RoboRigor pairs them episode by episode. The arm is encoded in
perturbation_axis/perturbation_level ("arm", e.g. "rho0", "rho0.05_q.pt", "bon4").

Usage: python scripts/phase2/to_roborigor.py --dumps outputs/phase2/dumps/base50 --arm rho0 \
           --results outputs/phase2/pi05_eval_libero_spatial_base50.json --out outputs/phase2/rr/base50
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def main(a):
    from roborigor.core.schema import EpisodeRecord

    res = json.load(open(a.results)) if a.results else {}
    suite = res.get("suite", a.suite)
    seed = int(res.get("seed", a.seed))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"records_{a.arm}_w0.jsonl"
    files = sorted(glob.glob(str(pathlib.Path(a.dumps) / "task*_ep*.npz")))
    assert files, f"no dumps in {a.dumps}"
    n = 0
    with open(path, "w") as f:
        for fn in files:
            z = np.load(fn, allow_pickle=False)
            tid, ep = int(z["task_id"]), int(z["episode_idx"])
            rec = EpisodeRecord(
                suite=suite, task_id=tid, task_description=str(z["task"]), init_id=ep, seed=seed,
                success=bool(z["success"]), wall_s=0.0, n_chunks=int(z["actions"].shape[0]),
                n_env_steps=int(z["n_env_steps"]), client_infer_s_total=0.0, server_infer_s_total=0.0,
                replan_steps=int(res.get("action_chunk", 5)), max_steps=MAX_STEPS[suite],
                policy_id="pi05_libero", checkpoint=str(res.get("checkpoint", a.checkpoint)), benchmark="libero",
                sampling_seed=seed, num_steps=int(res.get("num_steps", 5)), chunk_len=int(z["actions"].shape[1]),
                exec_horizon=int(res.get("action_chunk", 5)), perturbation_axis="arm", perturbation_level=a.arm,
                run_id=a.arm, started_at=str(res.get("started", "")),
            )
            f.write(rec.to_json() + "\n"); n += 1
    print(f"wrote {n} records -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dumps", required=True)
    p.add_argument("--arm", required=True, help="label, e.g. rho0, rho0.05, bon4")
    p.add_argument("--results", default="", help="results JSON from libero_eval_guided (for suite/seed/steps)")
    p.add_argument("--out", required=True)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--checkpoint", default="RLinf/RLinf-Pi05-LIBERO-SFT")
    main(p.parse_args())
