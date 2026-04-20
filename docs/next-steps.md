# Novel research directions — Nihal × Haotian, post-TFG/DDRL

Based on what the pilot experiments on SD3.5-M revealed (TFG gives +0.044 CLIP at
ρ=20, monotonic response; DDRL at our scale was a null; TFG gain invariant to
weight perturbations), here are four candidate directions worth pitching Haotian.
Each is sketched with (a) the claim, (b) why it's above the NeurIPS bar, (c) what
to build, (d) feasible compute.

Ranked by my estimate of impact × feasibility.

---

## 🥇 1. TFG Distillation — amortizing inference-time guidance into model weights

**Claim.** You can distill a TFG-augmented sampling trajectory back into the base
model's weights via supervised / score-matching losses. The distilled model
matches TFG's inference-time quality at **baseline inference cost**, and —
surprisingly — still benefits from TFG at test time (nonzero residual headroom).

### Why it clears the NeurIPS bar

- **Novel method.** TFG-to-weights distillation hasn't been published (as of our
  knowledge; needs a 30-min lit check). It connects three active threads:
  knowledge distillation, test-time-compute amortization (o1-style), and
  inference-guidance-aware fine-tuning. The ML community is actively asking
  "how do we turn inference compute into training compute?" — this is a clean
  answer for diffusion.
- **Resolves the complementarity question** we couldn't answer empirically.
  DDRL with CLIP reward didn't train in our pilot; distillation from a TFG
  rollout is a *much* stronger signal (the teacher's output *is already the
  direction we want*). If distilled + TFG-inference beats TFG-inference alone,
  that's the "RL + TFG are complementary" story — rigorously demonstrated.
- **Theoretical hook.** Can frame distillation as an M-step that amortizes the
  on-policy guidance gradient. Compare rates: full BPTT (DRaFT) vs REINFORCE
  (DDPO-style) vs distillation-from-guided-rollout. Our pilot showed DDRL's
  signal-to-noise is poor; distillation's teacher provides low-variance
  supervision.
- **Practical impact.** TFG at ρ=20 costs ~3× baseline inference wall. The
  distilled model is 1× at inference and retains the quality. Deployable.

### What to build

Two training schemes, both cheap:

**(a) Flow-matching supervised distillation.** Sample noise `ε`; run TFG-guided
denoising on the *frozen* base SD3.5-M to produce `x_0^TFG`; treat
`(ε, x_0^TFG, prompt)` as training data; train a LoRA (or full) finetune of the
base using the standard SD3 flow-matching loss. Because the target is already an
image the teacher produced, there's no reward variance — it's standard
supervised finetuning.

**(b) DRaFT-style on TFG trajectories.** Same as (a) but instead of minimizing
velocity MSE to the teacher's trajectory, maximize CLIP reward on the student's
own rollout, with the teacher's trajectory as an advantage estimator /
variance-reduction baseline. Combines best of both.

Evaluate the 2×3:

| | No TFG @ inference | TFG @ inference |
|---|---|---|
| Base SD3.5-M | Cell A (0.2681) ← our pilot | Cell B (0.3119) |
| TFG-distilled | *new* — C' | *new* — D' |

**Key experiments:**
1. Is C' > Cell A? (Did distillation work?)
2. Is D' > Cell B? (Do TFG + distilled weights stack?)
3. Is C' ≈ D'? (Did distillation fully absorb TFG?)
4. How does C' scale with distillation compute?

### Compute

- Distillation dataset generation: 1k–5k (prompt, TFG-image) pairs. ~2–8 hr on
  A100 with ρ=10. **~$4–16.**
- Distillation training (LoRA r=64, 2k–5k steps on the generated set): ~1–3 hr on
  A100. **~$2–6.**
- Full eval sweep (A, B, C', D' × multiple ρ): ~2 hr. **~$4.**
- **Total for a single clean run: ~$15–30.** Full paper probably 3–5x that for
  ablations (different teacher ρ, different distillation objectives, scaling).

### Risk

- If C' ≈ Cell A, distillation didn't work — probably a data-quantity issue,
  scalable by generating more pairs. Low risk.
- If C' ≈ Cell B (fully absorbed) AND D' ≈ C' (no residual), story is weaker
  ("TFG fully distillable, nothing new at inference"). Still publishable, but
  the exciting version requires a residual gap.

### Extension that would clinch it

Predictor-swap test: distill using CLIP-L as the TFG reward, then at inference
use a *different* predictor (SigLIP, aesthetic score) for new TFG. If the
distilled weights still transfer — TFG gains are predictor-agnostic, distillation
captures something general about the guidance direction. Nice ablation.

---

## 🥈 2. Video TFG-Flow — first principled demonstration on flow-matching video

**Claim.** Extend TFG to video flow-matching models (CogVideoX, Wan, HunyuanVideo)
with *temporally-aware predictors*, and characterize the gradient structure on
the time axis.

### Why it clears the NeurIPS bar

- **First demonstration** (modulo lit check — fast-moving space, check 2025-2026
  video-diffusion venues). Haotian's original TFG is image-only DDIM; our pilot
  extended to flow matching on images. Video is the obvious next frontier and
  is what Nihal pitched in the first email.
- **Genuinely new technical content**: the guidance gradient on a video latent
  has structure (per-frame components + cross-frame coherence components) that
  isn't present in image TFG. Per-frame guidance risks breaking temporal
  coherence; cross-frame guidance needs a predictor that scores the *clip*, not
  each frame independently.
- **Predictor design is a contribution by itself**: a temporally-aware reward
  like video-CLIP, optical-flow smoothness, or action-recognition probability
  over the clip. Each has tradeoffs worth ablating.

### What to build

1. Port `scripts/tfg_flow.py` to video pipelines (e.g., CogVideoX's
  `CogVideoXPipeline`). The scheduler conventions are the same (flow-matching
   Euler), but the VAE decode and transformer shapes change.
2. Implement 3 temporal predictors:
   - Video-CLIP (X-CLIP / VideoCLIP): scores the whole clip against prompt.
   - Frame-coherence: mean squared optical flow between consecutive frames
     (inverted — less motion = higher score, with saturation).
   - Action-alignment: CLIP-score on action labels derived from a frozen
     action-recognition model (e.g., VideoMAE).
3. Characterize the per-axis guidance gradient:
   - Dominant frame vs distributed gradient magnitude.
   - Temporal smoothness: does TFG on Frame 0 leak into Frame T?
4. Compare TFG alone vs RL fine-tuning (FlowGRPO for video if available) vs
   TFG + RL.

### Compute

This is the expensive one. Video models are 10–30× larger per inference than
SD3.5-M. CogVideoX-5B at 480p × 8 frames is ~30 s per generation on A100;
TFG doubles it. A proper 30-prompt sweep across 3 predictors × 4 ρ values is
30 × 3 × 4 × 60 s ≈ 6 hr generation alone. Plus eval.

- Realistic budget: **$50–150** on A100, single sweep.
- Full paper: likely $200–500+ for ablations. Non-trivial for a PhD.

### Risk

High. Temporal coherence may break catastrophically at useful ρ values — the
per-frame gradient can produce inconsistent sequences. This *is* the research
question, but if the answer is "TFG fundamentally can't work on video without
heavy re-engineering," that's a weaker paper.

### Why Haotian likely likes this

It's the direction Nihal originally pitched him. He greenlit images-first as a
warmup because he presumably agrees video is the interesting frontier. Showing
video TFG works, even modestly, is a high-value collaboration with him.

---

## 🥉 3. Non-differentiable-reward TFG via score-function gradient estimation

**Claim.** TFG can be extended to work with *non-differentiable* reward functions
(GenEval, WinoGround, user preferences) by replacing the pathwise gradient with
a REINFORCE-style score function estimator at each guidance step.

### Why it clears the NeurIPS bar

- **Opens up the predictor space drastically.** Currently TFG requires
  differentiable predictors — that's why everyone uses CLIP as a stand-in. With
  a REINFORCE-style estimator, you can guide on anything: GenEval's
  mmdetection detector, a code-executing function, binary user preferences, LLM
  judgments.
- **The "CLIP proxy" caveat** in every TFG-ish paper goes away. GenEval becomes
  a first-class guidance signal.
- **Theory carries over** from policy gradient literature: variance reduction
  via control variates, baselines, Rao-Blackwellization. Apply those tools to
  diffusion guidance.

### What to build

At each TFG step, replace the autograd gradient `∇_{x_t} log p(y | x_0(x_t))`
with a Monte Carlo estimate:

```
∇ log p ≈ E_ε [R(decode(x_0(x_t) + σε)) · ε / σ²]
```

This is the classic score-function estimator. Use control variates (e.g.,
CLIP-predictor-based as a low-variance baseline) to cut variance.

- Compare variance and final quality vs pathwise (CLIP-predictor) TFG.
- Show quality gain on GenEval by guiding on GenEval directly (breaks the
  CLIP-proxy caveat).
- Analyze variance reduction schemes.

### Compute

Cheap. Once the estimator is built, it's a drop-in replacement for the predictor
gradient. Main cost: more MC samples per step (eps_bsz in upstream TFG). **~$15–40**
for a clean paper.

### Risk

Variance may kill it. Diffusion guidance on an MC gradient may be too noisy to
help. Mitigations (control variates, learned baselines) are the technical
contribution. If the noise floor is too high, fallback to "GenEval-guided TFG
with variance-reduction tricks doesn't beat CLIP-guided TFG on GenEval-the-metric"
— a smaller but still publishable negative result.

---

## 🪧 4. Theoretical ρ schedule — optimal step-size from flow-matching SNR

**Claim.** The ρ response curve we observed (monotonic, ~log-linear) has a
closed-form explanation in terms of the predictor Lipschitz constant and the
per-step flow-matching signal-to-noise ratio. Derive the optimal ρ(t) schedule
from first principles and show it beats the flat-ρ and upstream-TFG heuristic
schedules.

### Why it clears the NeurIPS bar

- Theory papers land at NeurIPS when they **explain** an empirical mystery and
  produce an actionable rule. Our pilot already has the empirical mystery
  (what's the right ρ?).
- Builds on a lot of clean prior work: Karras et al. (EDM) for diffusion SNR,
  Chen et al. for posterior sampling theory, the classical control-theoretic
  view of guidance.
- Produces an off-the-shelf ρ schedule anyone running TFG can use. Utility
  signal for reviewers.

### What to build

- Derive: given a predictor with Lipschitz constant L and log-probability
  gradient norm bound G, find the ρ(t) that maximizes expected reward subject
  to a KL budget against the unguided distribution.
- Empirically validate: compare derived schedule against flat ρ, increasing,
  decreasing, on our 30-prompt setup.

### Compute

Minimal. Can reuse the existing ρ-sweep infrastructure. **<$10 for all experiments.**

### Risk

If the theoretical schedule doesn't beat empirical tuning, the paper is
"interesting theoretical framework, but practice still wins." Still publishable
but weaker.

---

## Recommendation

**Do #1 (TFG Distillation) first.** Cleanest story, most tractable, most
directly extends what we have, and specifically answers the complementarity
question we couldn't close with DDRL. Cost $15–30 for a first clean pass, ~2–3
weeks of work for a solid first draft.

If #1 produces strong results, you have leverage to ask Haotian for compute
support on #2 (video), which is the natural continuation.

#3 and #4 are parallel tracks — #3 has higher upside (opens the predictor
space) but higher variance; #4 is a clean theory-backed paper that could share
an author with #1 (Haotian would likely want to be on the theory paper).

## Concrete first step for Nihal

**Set up the TFG-distillation pipeline.** Code delta from what we have today:

1. `scripts/cloud/generate_tfg_dataset.py` — loop over prompts (larger pool,
   ~1k from DiffusionDB / GenEval / Nihal's pick), generate with TFG ρ=10 at
   512², save `(prompt, image, init_noise_seed)` triplets.
2. Extend `train_ddrl.py` with a `--distillation_target <path>` mode: instead
   of computing CLIP reward on the rollout, compute MSE (or flow-matching loss)
   against the saved TFG-image for that prompt.
3. Same `run_rho_sweep.py --checkpoint distilled` evaluation harness as today.

All three are <200 LOC of additional code. We already have the hardest pieces
(the TFG hook, the training loop, the evaluator).

**One-line pitch to Haotian**:

> I have TFG-Flow running cleanly on SD3.5-M (ρ=20 gives +4.4 CLIP points on 30
> GenEval prompts, monotonic response). Mini-DDRL with CLIP reward didn't move
> the model on our compute budget, so the "complementarity" question is still
> open. Next: distill TFG trajectories into base weights, which gives a
> low-variance training signal where DDRL/CLIP-reward was too noisy — and lets
> us test whether TFG provides *residual* gain after distillation. Cleanly
> addresses the inference-compute-amortization question.

Three short paragraphs when you're ready to send — happy to draft.
