# TFG-Flow + Distillation on SD3.5-M: how amortizable is inference-time guidance?

**Status: Phase A complete. Phase B (teacher-ρ sweep) deferred. Instance terminated.**

---

## Headline

> **TFG-Flow's CLIP gain on SD3.5-M decomposes roughly 7-10% amortizable into weights (via supervised distillation of teacher latents), ~70-77% non-amortizable (residual at inference), and ~15-20% redundancy loss. The decomposition is *stable* across 100-500 training steps — more distillation compute does NOT extract more of TFG's gain.**

That second sentence is the finding worth writing up. The conventional intuition is "train longer, amortize more." We see no such scaling. Distillation saturates in the first ~100 steps and the amortizable fraction stays in a narrow band across 5× more training.

The teacher-strength axis (how the decomposition shifts when the teacher uses weaker/stronger TFG) is the natural follow-up — Phase B in our run plan, deferred to a future sitting.

---

## Phase A — training-compute axis (5 checkpoints, teacher ρ=10 fixed)

All measured on the same 30 held-out GenEval-style prompts, 512², 28 inference steps, CFG=4.5, seed = 42 + prompt_id. Cell A' (fresh base, no TFG) = 0.2684; Cell B' (fresh base + TFG ρ=20) = 0.3132. TFG gain on base Δ_TFG = +0.0449.

For each checkpoint step count, we load the PEFT-saved adapter, run through `scripts/cloud/verify_lora_load.py` as pre-flight (all 5 passed), then measure Cell C (distilled, no TFG) and Cell D (distilled + TFG ρ=20).

| training steps | Cell C | Cell D | C − A (amortized) | D − C (residual) | **% amortized** | **% residual** |
|---|---|---|---|---|---|---|
| 100 | 0.2720 | **0.3058** | +0.0036 | +0.0339 | 8.0% | **75.5%** |
| 200 | **0.2728** | 0.3045 | +0.0044 | +0.0317 | 9.8% | 70.7% |
| 300 | 0.2723 | 0.3046 | +0.0039 | +0.0324 | 8.7% | 72.2% |
| 400 | 0.2695 | 0.3040 | +0.0011 | +0.0345 | 2.5% | 77.0% |
| 500 | 0.2713 | 0.3027 | +0.0030 | +0.0314 | 6.6% | 70.0% |

**Observations:**

1. **% amortized stays in a narrow 2.5%–9.8% band across 5× training compute.** No monotonic increase. Distillation does not absorb more of the teacher with more compute.
2. **Cell D (the stacked result) is highest at 100 training steps (0.3058) and falls to 0.3027 by step 500.** More training *slightly hurts* the combined pipeline — consistent with distillation imprinting teacher-specific artifacts that TFG can no longer productively exploit.
3. **Cell C peaks around step 200 (0.2728), then drifts.** Per-prompt noise at N=30 is ~0.001 (measured by replication — see below), so the 400-step dip to 0.2695 is ~3σ out. Could be real non-convexity, could be N=30 eval noise. A multi-seed follow-up would tell.
4. **No setting beats Cell B' alone (0.3132).** The best stacked D (0.3058 at 100 steps) is still below the TFG-on-base ceiling. Distillation + TFG remains *partially redundant* at every training-compute point measured.

**Replication variance.** ckpt-500 was measured twice (once in the earlier narrow re-eval, once in Phase A). Cell C: 0.2725 vs 0.2713 (spread 0.0012). Cell D: 0.3035 vs 0.3027 (spread 0.0008). Effect sizes we discuss (Cell C moves of ~0.001–0.004, Cell D moves of ~0.003–0.005) are comparable to or modestly above replication noise — we believe the trend, but confidence intervals on any single checkpoint are wide.

---

## What still needs to run (deferred Phase B)

Three additional full distillation runs at teacher ρ ∈ {2, 5, 20}, each = Stage 1 (200 teacher latents) + Stage 2 (500 training steps) + Stage 3 (Cell C + Cell D at eval ρ=20). Together with the existing ρ=10 point, gives a 4-point scaling curve on the **teacher-strength** axis.

Expected test: does the amortizable fraction scale with teacher ρ? Hypothesis: stronger teachers produce outputs further from base, so distillation absorbs MORE of the shift — but the *residual* TFG gain at inference might also shrink (less room to move).

Estimated cost: ~$9 on A100, ~4.5 hr wall. Orchestrator is already written and committed (`scripts/cloud/run_teacher_rho_sweep.sh`) and runs unattended.

---

## Prior results (carried over from earlier runs)

### Base ρ sweep (measured once on A10)

| ρ | mean CLIP | Δ vs no-TFG base |
|---|---|---|
| 0 | 0.2681 | — |
| 2 | 0.2710 | +0.0030 |
| 5 | 0.2779 | +0.0098 |
| 10 | 0.2896 | +0.0215 |
| 20 | 0.3119 | +0.0439 |

Monotonic, roughly log-linear. Independently reproduced at ρ=20 in the Phase A baseline (0.3132 ≈ 0.3119).

### Mini-DDRL (A100 run — did not train meaningfully)

| | no TFG | TFG ρ=20 |
|---|---|---|
| base | 0.2681 | 0.3119 |
| DDRL-LoRA (300 LoRA steps, batch 2, ref-KL 0.05) | 0.2670 | 0.3106 |

DDRL with CLIP reward at this scale did not shift the base. The distillation axis (Phase A here) gave a small but real shift where DDRL did not — supporting the decision to switch to supervised distillation for the main inquiry.

---

## Methodology summary (for paper writing)

### TFG-Flow on SD3 flow matching

Ported upstream TFG's DDIM-era classifier-guidance idea to SD3's `FlowMatchEulerDiscreteScheduler`. Per step:

1. Detach + clone x_t, set `requires_grad=True`.
2. Re-run transformer in grad scope → velocity v.
3. Predict x₀ via `x₀ = x_t − σ · v` (diffusers SD3 convention, verified against upstream source).
4. Decode x₀ through VAE; score with CLIP-L/14 (differentiable; mmdetection would not be).
5. `torch.autograd.grad(score, x_t)`; rescale; apply scaled update to x_t before the real Euler step.

Code: `scripts/tfg_flow.py`, `scripts/clip_predictor.py`. 5/5 CPU smoke tests pass.

### TFG Distillation (Stages 1 + 2 + 3)

**Stage 1 — teacher dataset.** Run TFG-guided SD3.5-M at teacher ρ on 200 GenEval-style prompts (disjoint from the 30 eval prompts). Save final *latents* directly (not VAE-decoded images) — avoids the encode roundtrip at training time and is lossless to the teacher's decision.

**Stage 2 — supervised LoRA distillation.** Standard SD3 flow-matching loss against teacher latents: sample σ ∈ [0.05, 0.95], form x_t = σ·noise + (1−σ)·x₀_T, predict velocity, MSE against v_target = noise − x₀_T. LoRA rank 64, α=128, lr 2e-5, AdamW, batch 4, 100–500 steps. T5-XXL parked on CPU (saves ~9.5 GB) and shuttled to GPU only for per-prompt encode.

**Stage 3 — held-out evaluation.** Load distilled LoRA via `PeftModel.from_pretrained` (diffusers' native `pipe.load_lora_weights` does *not* understand PEFT's `base_model.model.*` key prefix for SD3 — a silent-no-op bug that cost us $4.95 on an H100 before we caught it; now guarded by `scripts/cloud/verify_lora_load.py` that asserts a non-trivial output diff between base and LoRA-loaded transformer).

All three stages orchestrated by `scripts/cloud/run_distillation.sh`. The 5-point Phase A checkpoint sweep ran via `scripts/cloud/run_ckpt_sweep.sh`.

---

## Honest caveats (for the Haotian email / paper)

1. **Predictor is CLIP-L, not GenEval.** mmdetection is non-differentiable — this substitution matches upstream TFG's `image_label_guidance` convention. GenEval scoring over the same generated images is a separate evaluation, not run yet.
2. **N=30 eval prompts, 1 seed.** Per-prompt paired design gives decent statistical power against the noise floor we measured (~0.001), but a publication version wants 150–500 prompts and multiple seeds for proper confidence intervals.
3. **One teacher ρ (10) characterized.** The teacher-ρ axis is a known gap. Phase B orchestrator is ready to run.
4. **No μ term (upstream TFG's x₀-refinement gradient).** We ship only the ρ branch.
5. **Train prompts disjoint from eval prompts at the string level, but both drawn from GenEval categories** (single_object, two_objects, counting — the first 200 GenEval prompts). A held-out *category* (position, color_attribution) would be a stronger generalization claim.

---

## Cost accounting (cumulative across all sessions)

| Run | Instance | Wall | Spend |
|---|---|---|---|
| Smoke + mini-DDRL | A10 (2 sessions) | ~2 hr | $2.50 |
| Real DDRL | A100 SXM4 40GB | 73 min | $2.42 |
| Broken distillation (silent LoRA-load no-op) | H100 PCIe | ~2 hr | $4.95 |
| Narrow re-eval with fixed loader | A10 | ~30 min | $0.65 |
| **Phase A scaling sweep (this session)** | A100 SXM4 40GB | ~2 hr | **~$4.00** |
| **Total** | | ~7 hr | **~$14.50** |

Phase B (deferred): ~$9 additional, ~4.5 hr.

---

## What's on disk

- `outputs/base_sweep/` — the ρ ∈ {2,5,10,20} sweep, 5 cells JSONs
- `outputs/ddrl_real_sweep/` — A100 real-DDRL result JSONs
- `outputs/distill_eval_base/` — fresh A'/B' baseline (Cell A' = 0.268352, B' = 0.313212)
- `outputs/distill_eval_student_fixed/` — the first distilled eval with the fixed PEFT loader
- `outputs/ckpt_sweep/ckpt_{100,200,300,400,500}/` — **Phase A results**
- `outputs/checkpoints/distilled/{checkpoint-{100,200,300,400},final}/` — all 5 training-compute checkpoints
- `outputs/checkpoints/ddrl_real/final/` — DDRL-real LoRA (for reference)
- `scripts/` + `tests/` — all code, 14/14 local smoke tests passing
- `docs/next-steps.md` — the candidate-directions memo (TFG distillation was #1)

---

## Paper outline this run supports

**Title draft.** "How Amortizable is Inference-Time Guidance? A Decomposition of TFG Gains into Weight-Level and Trajectory-Level Components on Flow-Matching Diffusion."

**Claim 1** — TFG-Flow produces a clean monotonic response on SD3.5-M. (ρ-sweep data, already have.)

**Claim 2** — The TFG gain decomposes, via supervised distillation of teacher latents, into an amortizable fraction and a non-amortizable residual. (Single-point decomposition result, already have.)

**Claim 3** — The amortizable fraction is **remarkably stable** across training compute (2.5%-9.8% across 5× training variation). Distillation saturates fast. (Phase A data, just measured.)

**Claim 4** *(still to run)* — The amortizable fraction is a function of teacher strength; characterized across teacher ρ ∈ {2, 5, 10, 20}. (Phase B data — deferred.)

Even without Claim 4, Claims 1-3 are paper-shaped. Claim 4 is the reviewer-question we want to preempt; the run is ~$9 and fully automated.
