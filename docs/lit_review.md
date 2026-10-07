# Novelty Review: Guidance vs RL Post-Training for Reward Alignment of Flow Matching Models (SD3.5-M)

Bottom line up front: **no claim is fully preempted**, but the qualitative headline "inference-time methods can beat fine-tuning" is already published in weaker forms (DAS, FK Steering), and the bare facts "guidance stacks on fine-tuned checkpoints" (RLG, LiDAR, Ma et al.) and "T2I RL inherits reward blind spots" (GRPO-Guard, Hong et al.) exist in the literature. What is genuinely new across the board is the *controlled experimental machinery*: same base model, matched compute, multi-seed RL, budget crossover curves, landscape predictors, and external task-grounded verification. The paper survives if it is framed as the rigorous, controlled study the field has been asserting informally, and it dies as a review target if framed as discovering any of the qualitative directions.

## 1. Verdict Table

| Claim | Status | Reasoning |
|---|---|---|
| C1: TFG beats FlowGRPO per-reward, controlled, budget-matched | **PARTIALLY PREEMPTED** (DAS, FK Steering; qualitative direction only) | DAS shows SMC test-time alignment matches/beats DDPO-family fine-tuning; FK Steering shows a 0.8B steered model beating a 2.6B DPO fine-tune. Both are uncontrolled (different base models, no compute matching, single RL checkpoints, no seeds). The controlled per-reward comparison of gradient TFG vs GRPO on the same flow-matching backbone at matched budget is new. |
| C2: Training-budget crossover curve (RL log-linear, below guidance) | **NOVEL** | No prior work has any training-compute axis for diffusion RL vs a fixed guidance bar. Flow-GRPO, FK Steering, and Ma et al. supply the headline numbers to reconcile against, but no crossover curve exists anywhere. |
| C3: Seed lottery in small-scale diffusion RL (1 in 3 runs regresses) | **NOVEL** | No diffusion-RL paper reports multi-seed variance; Flow-GRPO, GRPO-Guard, DanceGRPO are all single runs. Only Henderson et al. 2017 established the phenomenon in classic deep RL; the diffusion-RL instantiation and its magnitude are new. |
| C4: Guidance dose-response slope predicts guidance-vs-RL margin | **NOVEL** (diagnostic tool not novel) | RSA-FT and Shen et al. use reward-landscape sharpness/smoothness as an explanatory variable, but only within one optimizer family. Using landscape steepness to predict which optimizer class wins is unprecedented. Drop any claim that landscape analysis per se is new. |
| C5: TFG composes with RL checkpoints (additive, rescues bad seeds); distillation does not | **PARTIALLY PREEMPTED** (RLG, LiDAR, FK Steering, Ma et al.; the additive half) | RLG improves GRPO checkpoints on SD3.5-M itself; LiDAR and FK Steering stack test-time methods on DPO fine-tunes; Ma et al. run search on DPO-SDXL. None stack an independent reward-gradient signal on online-RL checkpoints, none show bad-checkpoint rescue, none test distillation. The bare "guidance adds on top of tuning" punchline is gone; the orthogonal-signal, rescue, and distillation-negative results are new. |
| C6: Stronger optimizers inherit reward blind spots harder; reward-vs-task dissociation via external metrics | **PARTIALLY PREEMPTED** (GRPO-Guard, Hong et al., DAS, RSA-FT; the existence of dissociation) | GRPO-Guard documents proxy-up/quality-down in the exact Flow-GRPO-on-SD3.5-M setting; Hong et al. systematize blind spots in autoregressive T2I RL. What survives: the optimizer-strength dose-response (guidance vs RL comparative axis, which requires the guidance arm nobody else has) and task-grounded external verification (EasyOCR readability, SigLIP, open-vocab detector) rather than other preference models. Do not claim to discover reward hacking. |
| C7: Mu branch stronger and cheaper than rho on flow matching; branches compose ~2x | **NOVEL** (with mandatory differentiation) | TFG defined the branches but never ablated them head-to-head; MPGD showed x0-gradient cheapness on DDPM only, as an alternative not a complement; TFG-Flow ported TFG to flow matching but for molecules with one formulation and no branch comparison. The strength ranking, cost accounting, and composition result on T2I flow matching are all new. |
| C8: Guidance beats best-of-N at matched inference compute | **NOVEL, but with adverse prior evidence to rebut** | Ma et al. compare search variants only, no guidance. FK Steering reports the *opposite* ordering (FK/BoN beat gradient guidance) on SDv1.5 with a naive DPS-style baseline and unmatched compute. C8 must engage that result explicitly and explain the reversal (TFG mu branch on flow matching), or reviewers will flag the contradiction. |
| C9: Policy-KL coefficient non-monotone in stability; default beta pathological | **NOVEL** (thesis anticipated qualitatively) | Flow-GRPO ablates KL only on/off at one beta per task; GRPO-Guard argues KL is the wrong knob and trains with KL=0 but runs no sweep; RLG proves an inference-time-scale/KL equivalence but never trains at multiple betas. The beta sweep and non-monotone stability finding are new; cite all three as partial prior data points. |

No claim is fully scooped. The two closest calls are C1 (direction well established, controls new) and C5 (RLG is in the same setting with an adjacent headline).

## 2. Five Closest Works and How We Differ

**1. Flow-GRPO (NeurIPS 2025, arXiv 2505.05470).** The exact method and backbone we benchmark. It establishes only that GRPO improves flow models at ~24-GPU scale, with no guidance arm, no BoN, no seeds, no budget curves, no beta sweep. We differ by adding everything comparative: the guidance baseline it lacks, budget matching that reconciles our flat RL gains with their large ones, seed variance, and a beta sweep extending their binary KL ablation. This is our foil, not our scoop; its headline framing, KL ablation, and minimal-reward-hacking claim each need explicit engagement.

**2. DAS (ICLR 2025 Spotlight, arXiv 2501.05803).** Closest prior "test-time beats fine-tuning" result. Differences: SMC particle resampling vs gradient TFG; SD1.5 DDPO/AlignProp vs SD3.5-M FlowGRPO; no compute matching between training and inference cost; single RL runs; generalization measured against other learned reward models, not ground-truth task metrics. Frame C1 as "DAS suggested this informally; we test it rigorously."

**3. RLG (arXiv 2508.21016).** Shows an inference-time operation improving GRPO checkpoints on SD3.5-M itself, so it is the most urgent differentiation target for C5. Key distinction: RLG extrapolates the RL update direction (provably equivalent to lowering the effective KL coefficient), amplifying whatever the run learned, pathologies included; we stack an independent external reward gradient, which is why we can rescue bad seeds where extrapolation should amplify failure. Also cite its scale-equals-KL theorem as context for C9 while noting it cannot probe training dynamics.

**4. FK Steering (ICML 2025, arXiv 2501.06848).** Supplies the strongest uncontrolled inference-beats-fine-tuning evidence (C1), prior art for additive stacking on DPO fine-tunes (C5), and an adverse result for C8 (their gradient-guidance baseline loses to FK/BoN). Differences: particle methods on DDPM models vs gradient TFG on flow matching; off-the-shelf DPO checkpoints vs same-backbone matched-budget GRPO; their guidance baseline is a naive k=1 DPS-style implementation and the comparison is not compute-matched. C8 must rebut their ordering in-setting.

**5. GRPO-Guard (arXiv 2510.22319).** Documents implicit over-optimization in Flow-GRPO on SD3.5-M (proxy up, gold metrics down) and argues KL is the wrong stabilization knob. Differences: they diagnose and fix within training (ratio normalization, KL=0); we characterize how dissociation scales with optimizer strength using the guidance arm they lack, verify with task-grounded external tools rather than preference-model composites, and provide the quantitative beta sweep behind the KL thesis they assert qualitatively.

Honorable mentions requiring active differentiation: LiDAR (guidance-on-DPO stacking plus "improvements larger than fine-tuning alone" quote, C5/C8), TFG-Flow (first TFG-on-flow-matching port, C7), MPGD (x0-gradient cheapness on DDPM, C7), Hong et al. CVPR 2026 (systematic T2I RL blind spots, C6), RSA-FT (sharpness as hacking predictor, C4).

## 3. Positioning Advice

**Lead with the controlled experiment, not the direction.** The paper's identity should be "the first controlled, compute-matched, multi-seed comparison of optimizer classes for reward alignment on flow matching," not "guidance beats RL." The latter claim, stated baldly, gets three citations thrown at it (DAS, FK Steering, LiDAR) in the first review. The former is defensible against everything found.

**Make C2 and C3 the headline contributions.** They are the cleanest fully-novel results and they are what makes C1 believable: the crossover curve explains how our results and Flow-GRPO's headline gains coexist (budget regimes), and the seed lottery explains why single-run comparisons in prior work are uninformative. C4 is the most intellectually distinctive claim and has no real precedent as a comparative predictor.

**Reconcile, do not just contradict, Flow-GRPO.** Reviewers will include its authors' orbit. Explicitly show where on the budget axis their 24-A800 results live relative to our 1x-8x single-GPU curve, and present C1/C2 as delimiting the regime where RL pays off rather than declaring RL loses.

**Reframe C5 defensively before submission.** RLG is in our exact setting with an adjacent headline. Rewrite C5 around three things RLG cannot do: orthogonal external signal (not KL-knob extrapolation), rescue of regressed seeds (extrapolation should worsen them, a testable contrast worth one experiment if cheap), and the distillation negative result.

**Pre-empt the FK Steering contradiction on C8.** State their guidance-loses-to-BoN result up front and explain the reversal mechanistically via C7 (mu branch on flow matching vs naive DPS on DDPM) and via matched compute. Turning an adverse result into an explained reversal is stronger than hoping nobody notices.

**Drop all "first to show" language on reward hacking (C6) and landscape analysis (C4).** Instead: "prior work established dissociation exists (GRPO-Guard, Hong et al.) and that sharpness predicts hacking within fine-tuning (RSA-FT); we show dissociation scales with optimizer strength and that landscape steepness predicts the margin between optimizer classes." Hong et al. is ammunition, not a threat: independent confirmation in a different model family.

**C7 framing:** "we ablate and quantify TFG's design space on flow matching," citing TFG for the branches, MPGD for x0-cheapness precedent on diffusion, TFG-Flow for the flow-matching port. Novelty is the strength ranking, cost accounting, and composition, positioned as quantitative confirmation-plus-extension of TFG's complementarity conjecture.

**De-emphasize:** any suggestion that guidance is universally superior (C2's crossover implies RL wins at some budget), and any framing of C9 as overturning Flow-GRPO rather than refining its binary ablation into a sweep.

## 4. Must-Cite List by Section

**Introduction / headline framing (C1, C2):**
- Flow-GRPO (2505.05470) - the method and gains we benchmark against
- DAS (2501.05803) - closest prior test-time-beats-fine-tuning evidence
- FK Steering (2501.06848) - uncontrolled inference-vs-fine-tuning wins
- Ma et al. inference-time scaling (2501.09732) - inference-compute-as-axis framing
- Snell et al. test-time compute (2408.03314) - LLM-side precedent for the crossover framing

**Method / TFG on flow matching (C7):**
- TFG (2409.15761) - source of the mu/rho design space
- TFG-Flow (2501.14216) - first TFG-on-flow-matching port; differentiate on branches, cost, composition, domain
- MPGD (o3BxOLoxm1) - x0-gradient cheapness precedent on DDPM
- Shen et al. (2403.12404) - why the x0 branch is better behaved
- Background guidance lineage: DPS (2209.14687), FreeDoM (2303.09833), LGD (song23k), Universal Guidance (2302.07121), FlowChef (2412.00100), Elucidating classifier guidance (2310.11311), Gradient Guidance optimization perspective (2404.14743), Not All Prediction Targets (2607.00647), Symplectic Adjoint (2312.12030)

**RL baselines and training setup (C1, C3, C9):**
- Flow-GRPO (2505.05470), DDPO (2305.13301), DRaFT (2309.17400), DanceGRPO (2505.07818), Pref-GRPO (2508.20751)
- Henderson et al. (1709.06560) - seed-variance precedent in deep RL (C3)
- GRPO-Guard (2510.22319) - KL-is-wrong-knob thesis (C9)
- RLG (2508.21016) - guidance-scale/KL equivalence theorem (C9)

**Composition experiments (C5):**
- RLG (2508.21016) - primary differentiation target
- LiDAR (2602.03211) - guidance-on-DPO stacking
- FK Steering (2501.06848) - Table 2 stacking evidence
- Ma et al. (2501.09732) - search-on-DPO-SDXL
- Noise Hypernetworks (2508.09968) - amortization/distillation context

**Reward hacking / dissociation (C4, C6):**
- GRPO-Guard (2510.22319) - dissociation in our exact setting
- Hong et al. (2601.03468) - systematic blind spots in T2I RL
- RSA-FT (2603.21175) - sharpness predicts hacking within fine-tuning
- Gao et al. overoptimization scaling laws (2210.10760)
- DAS (2501.05803) - over-optimization contrast, MIRA (2510.01549), GARDO (2512.24138), Shen et al. (2403.12404) - adversarial-gradient mechanism

**Best-of-N / inference scaling (C8):**
- Ma et al. (2501.09732) - canonical BoN scaling curves, the baseline C8 argues against
- FK Steering (2501.06848) - adverse ordering that must be engaged
- SVDD (2408.08252), Sampling Demons (2410.05760), LiDAR (2602.03211) - efficiency framing overlap

**Related-work surveys / conceptual bridges:**
- Uehara et al. tutorial (2501.09685), RL-guidance-scaling anthology (2509.04372), Scaling Scaling Laws with Board Games (2104.03113) - train-vs-test compute trade precedent

## 6. Addendum (2026-10-05): papers since the August review that touch the new claims

Searched for: reward-gradient fine-tuning vs guidance comparisons, reward hacking in gradient
fine-tuning, diagnostics predicting when RL vs guidance helps, guidance distillation.

| Paper | Date | What it does | Threat to us | How we position |
|---|---|---|---|---|
| FlowBP: Exploring the Design Space of Reward Backpropagation for Flow Matching (arXiv 2606.11075) | Jun 2026 | Engineers the backward trajectory of reward backprop (sparse Euler, bridge coupling, Lagrange quadrature) to bound memory and gradient inflation; SD3.5-M, FLUX.1-dev, FLUX.2 | Reviewer: "your DRaFT-1 is the naive gradient trainer; use FlowBP" | Cite as the strongest weights×gradient method; DRaFT-1 has one Jacobian step so gradient inflation is not our failure, reward hacking is. Run FlowBP as an arm if code is usable. No GRPO or guidance comparison, no hacking analysis in it. |
| Reward Sharpness-Aware Fine-Tuning (RSA-FT, arXiv 2603.21175) | Mar 2026 | Reward hacking as an adversarial attack on a sharp reward model; input/weight perturbations as defense | None to the comparison; it explains our collapse | Cite as the known phenomenon. Ours: the controlled contrast that GRPO on the same reward/budget does not hack while the gradient trainer does (the signal axis). |
| Reward Score Matching (arXiv 2604.17415; ICML'26 SPIGM best paper) | Apr 2026 | Unifies weight-side reward fine-tuning as score matching against a value-guided target; bias/variance/compute tradeoffs | Framing overlap on the weights side | Cite; it unifies one row of our 2×2. We compare across the location axis it does not cover, at matched compute, with seeds and audits. |
| Reinforce Adjoint Matching (RAM, arXiv 2605.10759) | May 2026 | Consistency-loss RL post-training; claims Flow-GRPO's peak reward in up to 50× fewer steps on SD3.5-M | Real: moves the budget crossover and weakens "RL is slow / a lottery" | Scope every RL claim to the method and budget tested; add RAM as an arm if feasible (cheap trainer); report crossover per method. |
| Flow Matching Policy Gradients (arXiv 2507.21053), TempFlow-GRPO (2508.04324), Stitched Value Model (2605.19804), Gradient-Free Noise Optimization (2605.11347) | 2025–26 | Newer RL / noise-space variants | Minor; more competitors on the weights×scalar and sampler×scalar cells | Cite in related work; none is a controlled cross-family comparison. |

Still unclaimed after this pass: the two-coordinate pre-flight diagnostic (slope × cross-prompt
gradient cosine) and a matched-budget, multi-seed comparison across all four cells of
{weights, sampler} × {scalar, gradient}.
