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
