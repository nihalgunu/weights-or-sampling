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

### Pre-launch diagnostics for the guided arms (Nov 4-5, all offline on base50 dumps, no GPU)
Scripts: scripts/phase2/diag/ (calibrate_rho.py uses the shipped Qs; grad_agree.py and grad_dims.py take a
directory holding q/{hi,lo}_s{0..4}.pt, five seeds per cross-fit half, trained with
`train_q.py --train_on final --episodes 25:50|0:25 --seed S`, 11 s each on CPU).
1. **The Q gradient is steep and consistent, not noise.** Raw gradient norm ~46 logits per unit action.
   Cosine between seeds trained on the same half 0.83; between ensembles trained on disjoint halves 0.62
   (all tasks 0.58-0.66). A step along one Q's gradient raises the other seeds almost as much as itself
   (+3.9 vs +3.5 logits at rho 0.005); a random direction of the same length gives +0.02.
2. **In raw action units the gradient is 99.6% rotation.** Energy share by dim (x y z rx ry rz grip) =
   .001 .001 .002 .306 .584 .106 .000. The rotation commands barely vary in the data (sd 0.03, 0.02, 0.06 vs
   0.45, 0.35, 0.35 for translation, 1.0 for the gripper) and Q standardizes its inputs, so a small raw move
   on a rotation dim is a large move for Q. The raw-metric sampler therefore pushes only rotation.
   Fix added: `--metric std` (GuidanceConfig.metric): steepest ascent measured in units of each dim's sd
   (step proportional to sd^2 * grad); CPU test 4d. In that metric the step is spread evenly over all 7 dims.
3. **Only the last guidance step survives.** In the toy-Q run (rho 0.05) the actual action shift at the first
   inference is 1.14x the last step's delta and 0.23x the sum of the five deltas (cosine 0.998). The policy is
   near-deterministic given the observation (rms(first x0_hat - final) = 0.054), so each Euler step pulls the
   sample back to the same x0 and erases earlier pushes. Guidance on this policy is, in effect, one gradient
   step on the final action.
4. **Calibration (last step only, held-out Q):** raw rho 0.002 / 0.005 / 0.01 -> logit +1.6 / +3.9 / +7.3 and a
   rotation shift of up to 0.4 / 0.9 / 1.9 sd. std rho 0.05 / 0.1 / 0.2 -> +1.7 / +3.4 / +6.6 and 0.1 / 0.2 /
   0.45 sd on every dim. For scale, swapping in another episode's chunk changes the logit by ~2.4. The earlier
   sweep {0.005 .. 0.05} raw would have been 1 to 10 sd on rotation.
5. **Pairing is exact.** The M1 unguided rerun (other A10 box, task subset 0,5,9) reproduces the base50
   episodes bit for bit over whole episodes (9 of 9). Any difference in a guided arm is caused by the arm.
   Keep every arm on an A10.

Arms to run (scripts/phase2/launch_guided_arms.sh): raw rho 0.002 and 0.005, std rho 0.05 and 0.1 (matched
logit gain across the two metrics), best-of-8 by Q. Tasks 0,5,3,6,9 x 50 trials (250 episodes): task 0 is the
harm control, the other four hold 99 of the 116 base50 failures. A10 is now $1.29/h (was $0.75), so the full
500 would be ~$7 per arm; the 250-episode subset is ~3.2 h, ~$4 per arm, ~$20 for five. NOT launched yet.
