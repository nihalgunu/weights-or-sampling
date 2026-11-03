# Phase 2: Weights or Sampling on a flow policy (pi0.5 / LIBERO)

Started Oct 29 2026. Phase 1 (SD3.5-M, 8 rewards) is closed; see email_update_oct2026.md.

## Question
Same question as Phase 1, on a robot policy: given a reward and a flow-matching policy,
is the reward better spent training the weights (online RL) or steering the sampler
(guidance / best-of-N)? Phase 1 says: guidance wins when the reward is differentiable and
steep, RL only draws level on learned preference rewards. H2 for policies: the gap is
*larger*, because the policy's reward (task success) is non-differentiable and the only
differentiable surrogate is a learned Q, which should be a worse guidance signal than CLIP.

## Model choice (checked Oct 29 2026)
- Newest Physical Intelligence models (pi0.6, pi0.7, Apr 2026) are closed; API waitlist only.
- Newest *open* weights: **pi0.5** (openpi, Apache-2.0, PyTorch port validated on LIBERO).
  So "update to the newest pi" = pi0.5. Nothing newer can be fine-tuned or steered by us.
- Base policy: `RLinf/RLinf-Pi05-LIBERO-SFT` (few-shot SFT pi0.5, PyTorch safetensors, 7.5 GB).
  LIBERO success Spatial/Object/Goal/Long = 84.6 / 95.4 / 84.6 / 43.9, avg 77.1.
  Headroom exists (the openpi full-shot pi05_libero checkpoint is at 96.9 and useless for us).
- RL recipe and code: piRL (arXiv 2510.25889) in RLinf. Their numbers after RL: GRPO 91.5, PPO 97.9.
  No RL-finetuned LIBERO checkpoints are released, only SFT, so the RL arm must be trained by us.
- Sampler: openpi PyTorch `sample_actions` is a 10-step Euler loop under `torch.no_grad`
  (RLinf eval uses num_steps 5, action chunk 5 of a 10-step horizon, action_dim 7). Guidance
  hooks there: `x_t = x_t + dt * (v_t - rho * grad_x Q(obs, x0_hat(x_t)))`.

## Arms (matched wall-clock on one GPU, same reward, same base policy)
1. **Base**: SFT policy, 5 denoising steps.
2. **RL (weights)**: piRL GRPO or PPO from the SFT checkpoint, budget B hours on one GPU
   (RLinf `libero_<suite>_ppo_openpi_pi05`, quickstart/offload settings).
3. **Guidance (sampling)**: TFG on the action-chunk denoising loop with a learned Q(obs, chunk)
   as the differentiable reward; rho swept; no weight change.
4. **Best-of-N by Q** (sampling, gradient-free): sample N chunks, execute argmax Q.
5. **Stacked**: RL policy + guidance.
Reward: Q trained once on SFT-policy rollouts (success-labelled Monte-Carlo returns), frozen,
shared by arms 3-5. RL uses true success (its natural reward). Audit = true success rate on
held-out initial states, 50 trials x 10 tasks per suite. Primary suites: Spatial (headroom 15 pts)
and Long (headroom 56 pts).

## Milestones
- **M0 (this week, ~$5-10):** reproduce SFT eval on LIBERO-Spatial on one A100 with RLinf's
  standalone script, in their Docker image. Target 84.6 +- 3. Runner: scripts/phase2/run_pi05_eval.sh.
- **M1 (~$10):** guidance hook with rho=0 reproduces M0 exactly; dump (obs, chunk, return) rollouts.
- **M2 (~$20):** train Q on the dumps; Best-of-N-by-Q number (first sampling-side result).
- **M3 (~$30):** TFG with Q, rho sweep, Spatial + Long.
- **M4 (~$100-200):** piRL GRPO/PPO at matched budget, 2-3 seeds; stacked arm.
- **M5:** write-up; Phase 1 + Phase 2 tables side by side.

## Risks
- RLinf Docker image + LIBERO EGL on a Lambda box: unknown until M0 runs.
- HF checkpoint layout vs openpi `load_norm_stats` (needs assets/<asset_id>/norm_stats.json).
- piRL PPO assumes multi-GPU; single-GPU matched budget may need GRPO and small env counts.
- Q quality bounds the guidance arm. That is the finding, not a bug, but it needs 2 Q seeds.

## Statistics design (added Nov 3, using RoboRigor)

Every arm runs the full suite at 50 trials/task (500 episodes) with the same `--seed`, so episodes
pair across arms by (task, init state, noise stream). Comparisons use `roborigor report` (exact
McNemar on the paired discordant counts, Wilson intervals, Holm across tasks); the eager hook at
rho=0 is the baseline arm, never the torch.compiled sampler (compiled-vs-eager differs by up to
0.21 in normalized action units). Informative tasks declared in advance from the smoke test:
task 5 (bowl on the ramekin, 4/10) and task 9 (bowl on the wooden cabinet, 3/10); the other eight
sit near ceiling and mostly certify "no change". Power (alpha .05, power .8): unpaired, 500/arm
detects ~7 points at base 0.85 and ~8 points at base 0.35; paired, 500 pairs detect a 5-point net
gain when discordance is 5%/10%. Tooling: scripts/phase2/to_roborigor.py converts dumps to
EpisodeRecord JSONL (`outputs/phase2/rr/<run>/`); monitor_jobs.sh / abort_job.sh give early stop.

### Baseline result and Q diagnostics (Nov 3)
- base50 (eager hook, rho=0, seed 7, 50 trials x 10 tasks): 384/500 = 76.8% (M0 compiled sampler gave 77/100). Per task:
  0:49 1:47 2:49 3:41 4:45 5:20 6:34 7:50 8:43 9:6. Equivalence vs eager 0.0 on 20 checks; compiled-vs-hook max 0.014.
- Informative tasks are now 3 (82%), 5 (40%), 6 (68%); 0/2/7 are at ceiling and 9 is near floor (12%), so the
  earlier "5 and 9" choice is superseded. Paired power at 500 pairs: +5 net at 10-15% discordance -> 0.80-0.94.
- Q (MLP on state[:8] + pooled prefix + executed chunk, BCE on episode success), 5 random episode splits:
  per-task held-out AUC 0.84 mean; action-swap AUC 0.73 (chunk carries signal), context-swap 0.61; counterfactual
  pair accuracy (same context, success chunk vs failure chunk) 0.75 on tasks 1/4/5, ~0.52 on 6/8.
- Cross-fitting for the guided arms: outputs/phase2/q/xfit_hi (trained on episodes 25-49) is applied to episodes 0-24
  and xfit_lo to 25-49 (`--q_ckpt xfit_hi/q.pt,xfit_lo/q.pt`), so no episode is steered by a Q that saw its init state.
  Held-out-half pair accuracy 0.65 / 0.62, task 5: 0.76 / 0.74.
