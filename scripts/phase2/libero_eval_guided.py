"""LIBERO eval for pi0.5 with the training-free guidance hook, rollout dumps, and a
built-in equivalence check against openpi's own sampler.

Mirrors RLinf's ``toolkits/standalone_eval_scripts/openpi/libero_eval.py`` main loop
(env construction, warm-up steps, image flip, state layout, replanning every
``--action_chunk`` steps) and adds:

* ``--sampler hook``: route ``Policy.infer`` through ``GuidedSampler``;
  ``--rho 0`` reproduces the base sampler bitwise (``--equivalence_checks N`` proves it
  in-process on the first N inferences by running both and comparing).
* Deterministic noise: every inference gets ``torch.randn`` from a CPU generator seeded
  by ``(seed, task_id, episode_idx, inference_idx)`` and passed to ``Policy.infer``,
  so arms differ only by the sampler. (The upstream script leaves torch unseeded.)
* ``--dump DIR``: per-episode ``.npz`` with model-space inputs/outputs for training Q.
* ``--results_json``: per-task counts written after every task.

Runs inside the RLinf Docker image with ``PYTHONPATH`` containing ``/workspace/RLinf``
and this directory.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time as _time

import imageio
import numpy as np
import torch
import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from toolkits.standalone_eval_scripts.openpi import setup_logger, setup_policy  # noqa: E402
from toolkits.standalone_eval_scripts.openpi import libero_eval as base  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from guided_sampler import GuidanceConfig, GuidedSampler, install  # noqa: E402

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def toy_q(name):
    """Hand-written Q for plumbing smokes (no learned model needed)."""
    if name == "none":
        return None
    if name == "small_action":  # prefers small normalized actions on the 7 real dims
        return lambda ctx, x: -(x[..., :7] ** 2).flatten(1).mean(1)
    if name == "smooth":  # prefers temporally smooth chunks
        return lambda ctx, x: -((x[:, 1:, :7] - x[:, :-1, :7]) ** 2).flatten(1).mean(1)
    raise ValueError(name)


def load_q(path, device):
    from q_model import load_q as _load  # written in M2

    return _load(path, device)


def noise_for(seed, task_id, ep, k, H, A):
    g = torch.Generator().manual_seed((((seed * 1000003 + task_id) * 1009 + ep) * 10007 + k) % (2**63 - 1))
    return torch.randn(1, H, A, generator=g, dtype=torch.float32).numpy()


def main(args):
    logger = setup_logger(args.exp_name, args.log_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps = MAX_STEPS[args.task_suite_name]
    task_ids = list(range(task_suite.n_tasks)) if not args.task_ids else [int(t) for t in args.task_ids.split(",")]
    logger.info(f"Task suite: {args.task_suite_name}; tasks {task_ids}; sampler={args.sampler} rho={args.rho}")

    logger.info("policy setup start")
    policy = setup_policy(args)
    logger.info("policy setup done")
    model = policy._model  # noqa: SLF001
    H, A = model.config.action_horizon, model.config.action_dim
    device = policy._pytorch_device  # noqa: SLF001

    sampler = None
    if args.sampler == "hook":
        q_fn = load_q(args.q_ckpt, device) if args.q_ckpt else toy_q(args.q_toy)
        cfg = GuidanceConfig(rho=args.rho, apply_range=(args.apply_lo, args.apply_hi),
                             clip_scale=args.clip_scale, grad_steps=args.grad_steps)
        sampler = GuidedSampler(model, q_fn=q_fn, cfg=cfg, record=bool(args.dump),
                                equivalence_checks=args.equivalence_checks)
        install(policy, sampler)
        logger.info(f"hook installed: {cfg} q={'ckpt:' + args.q_ckpt if args.q_ckpt else 'toy:' + args.q_toy} "
                    f"equivalence_checks={args.equivalence_checks} dump={args.dump}")
    elif args.sampler != "plain":
        raise ValueError(args.sampler)

    dump_dir = pathlib.Path(args.dump) if args.dump else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    results = {"suite": args.task_suite_name, "trials_per_task": args.num_trials_per_task, "checkpoint": args.pretrained_path,
               "num_steps": args.num_steps, "action_chunk": args.action_chunk, "seed": args.seed, "sampler": args.sampler,
               "rho": args.rho, "apply_range": [args.apply_lo, args.apply_hi], "clip_scale": args.clip_scale,
               "grad_steps": args.grad_steps, "q": args.q_ckpt or args.q_toy, "task_ids": task_ids,
               "per_task": [], "total_successes": 0, "total_episodes": 0, "total_success_rate": None,
               "equivalence": None, "started": _time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None}

    def write_results():
        if sampler is not None and sampler.equivalence_results:
            er = sampler.equivalence_results
            results["equivalence"] = {"n": len(er), "max_abs_diff": max(er), "n_nonzero": sum(1 for d in er if d != 0.0)}
        if args.results_json:
            tmp = args.results_json + ".tmp"
            json.dump(results, open(tmp, "w"), indent=1)
            os.replace(tmp, args.results_json)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = base._get_libero_env(task, base.LIBERO_ENV_RESOLUTION, args.seed)  # noqa: SLF001

        task_episodes, task_successes = 0, 0
        for episode_idx in range(args.num_trials_per_task):
            logger.info(f"\nTask: {task_description}")
            logger.info(f"Starting episode {task_episodes + 1}...")
            policy.reset()
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            replay_images, exec_chunks, step_ts = [], [], []
            if sampler is not None:
                sampler.records.clear()
            k = 0  # inference index within the episode
            done = False
            for t in range(max_steps + args.num_steps_wait):
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(base.LIBERO_DUMMY_ACTION)
                    continue
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                replay_images.append(img)
                state = np.concatenate((obs["robot0_eef_pos"], base._quat2axisangle(obs["robot0_eef_quat"]),  # noqa: SLF001
                                        obs["robot0_gripper_qpos"]))
                if not action_plan:
                    observation = {"observation/image": img, "observation/wrist_image": wrist_img,
                                   "observation/state": state, "prompt": str(task_description)}
                    noise = noise_for(args.seed, task_id, episode_idx, k, H, A)
                    action_chunk = policy.infer(observation, noise=noise)["actions"]
                    k += 1
                    assert len(action_chunk) >= args.action_chunk, (
                        f"We want to replan every {args.action_chunk} steps, but policy only predicts {len(action_chunk)} steps.")
                    action_plan.extend(action_chunk[: args.action_chunk])
                    exec_chunks.append(np.asarray(action_chunk, dtype=np.float32))
                    step_ts.append(t)
                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
            task_episodes += 1
            total_episodes += 1

            if dump_dir is not None and sampler is not None and sampler.records:
                R = sampler.records
                np.savez_compressed(
                    dump_dir / f"task{task_id:02d}_ep{episode_idx:03d}.npz",
                    task_id=task_id, task=task_description, episode_idx=episode_idx, success=bool(done),
                    n_env_steps=t + 1, step_t=np.asarray(step_ts),
                    noise=np.stack([r.noise[0].numpy() for r in R]),
                    state=np.stack([r.state[0].numpy() for r in R]),
                    prefix_pool=np.stack([r.prefix_pool[0].numpy() for r in R]),
                    x0_hats=np.stack([torch.stack(r.x0_hats)[:, 0].numpy() for r in R]),
                    times=np.asarray(R[0].times, dtype=np.float32),
                    actions=np.stack([r.actions[0].numpy() for r in R]),
                    exec_actions=np.stack(exec_chunks),
                    q_values=np.asarray([r.q_values for r in R], dtype=np.float32) if R[0].q_values else np.zeros(0),
                )
                sampler.records.clear()

            if total_episodes <= args.num_save_videos:
                suffix = "success" if done else "failure"
                out_path = pathlib.Path(f"{args.log_dir}/{args.exp_name}/") / f"rollout_{task_description.replace(' ', '_')}_{episode_idx}_{suffix}.mp4"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(out_path, [np.asarray(x) for x in replay_images[:: args.video_temp_subsample]],
                                 fps=30 // args.video_temp_subsample)

            logger.info(f"Success: {done}")
            logger.info(f"# episodes completed so far: {total_episodes}")
            logger.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            if sampler is not None and sampler.equivalence_results:
                logger.info(f"equivalence so far: n={len(sampler.equivalence_results)} max_abs_diff={max(sampler.equivalence_results):.3e}")
        env.close()

        rate = task_successes / task_episodes if task_episodes else 0.0
        results["per_task"].append({"task_id": task_id, "task": task_description, "succ": task_successes, "n": task_episodes, "rate": rate})
        results["total_successes"], results["total_episodes"] = total_successes, total_episodes
        results["total_success_rate"] = total_successes / total_episodes if total_episodes else None
        write_results()
        logger.info(f"Task: {task_description}, Successes: {task_successes}/{task_episodes}, Success Rate: {rate:.2%}")

    results["finished"] = _time.strftime("%Y-%m-%d %H:%M:%S")
    write_results()
    logger.info("\n===============")
    for r in results["per_task"]:
        logger.info(f"{r['task']}: {r['rate']:.2%}")
    logger.info(f"\nTotal Success Rate: {total_successes}/{total_episodes} = {results['total_success_rate']:.2%}")
    logger.info(f"results/total_success_rate: {results['total_success_rate']}")
    logger.info(f"results/total_episodes: {total_episodes}")
    logger.info(f"results/total_successes: {total_successes}")
    if results["equivalence"]:
        logger.info(f"results/equivalence: {results['equivalence']}")
        if args.equivalence_strict and results["equivalence"]["n_nonzero"]:
            logger.error("equivalence check FAILED: hook differs from base sampler")
            sys.exit(3)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--exp_name", default="libero_spatial_pi05_guided")
    p.add_argument("--config_name", default="pi05_libero")
    p.add_argument("--pretrained_path", default=None)
    p.add_argument("--task_suite_name", default="libero_spatial")
    p.add_argument("--num_trials_per_task", type=int, default=50)
    p.add_argument("--action_chunk", type=int, default=5)
    p.add_argument("--num_steps", type=int, default=5)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--num_save_videos", type=int, default=5)
    p.add_argument("--video_temp_subsample", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--task_ids", default="", help="comma list; default all tasks in the suite")
    # guidance
    p.add_argument("--sampler", choices=["plain", "hook"], default="hook")
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--apply_lo", type=float, default=0.0)
    p.add_argument("--apply_hi", type=float, default=1.0)
    p.add_argument("--clip_scale", type=float, default=1.0)
    p.add_argument("--grad_steps", type=int, default=1)
    p.add_argument("--q_ckpt", default="")
    p.add_argument("--q_toy", default="none", choices=["none", "small_action", "smooth"])
    # instrumentation
    p.add_argument("--equivalence_checks", type=int, default=0)
    p.add_argument("--equivalence_strict", action="store_true")
    p.add_argument("--dump", default="")
    p.add_argument("--results_json", default="")
    main(p.parse_args())
