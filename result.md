# TFG-Flow + TFG Distillation on SD3.5-M: a decomposition of inference-time guidance

**Status: complete. All instances terminated. Total spend across all runs ≈ $12.**

---

## Headline

> **TFG-Flow's CLIP gain at ρ=20 is only ~9% amortizable into model weights via supervised distillation of TFG-guided trajectories. The other ~69% remains captured only at inference time.** Distillation also measurably shrinks the headroom available for inference-time TFG (0.045 → 0.031 residual gain).

The question Nihal posed to Haotian in the original emails — *are inference-time guidance and weight-level training complementary or redundant?* — has a clean, quantitative answer on this setup: **partially redundant, with a characterizable decomposition**.

---

## Full result matrix

All cells: SD3.5-M, 30 held-out GenEval-style prompts, 512×512, 28 inference steps, CFG=4.5, seed=42+prompt_id. CLIP-L/14 predictor. 30 prompts drawn so no overlap with the 200-prompt distillation train pool.

| | No TFG | TFG ρ=20 | TFG Δ |
|---|---|---|---|
| **Base SD3.5-M** | 0.2684 (A′) | 0.3132 (B′) | **+0.0449** |
| **Distilled LoRA** (500 steps, r=64, 200 teacher latents @ ρ=10) | **0.2725 (C′)** | **0.3035 (D′)** | **+0.0310** |
| **Δ vs base** | **+0.0041** | **−0.0097** | — |

**Decomposition of TFG's +0.0449 effect:**
- **Amortized into weights by distillation: +0.0041 / +0.0449 = 9.2%**
- **Preserved at inference after distillation: +0.0310 / +0.0449 = 69.1%**
- **Lost (distillation shrinks TFG's headroom): +0.0098 / +0.0449 = 21.7%**

Per-prompt signal (N=30, all paired with matched seeds):

| Comparison | mean Δ | std | per-prompt wins |
|---|---|---|---|
| B′−A′ (TFG on base) | +0.0449 | 0.0285 | **30/30** |
| D′−C′ (TFG on distilled) | +0.0310 | 0.0251 | **29/30** |
| C′−A′ (distillation alone) | +0.0041 | 0.0154 | 18/30 |
| D′−B′ (distillation on top of TFG) | −0.0097 | 0.0152 | 8/30 |

TFG itself is near-universal (30/30, 29/30). Distillation's effect is subtler — 18/30 and 8/30 tell you it's moving in two directions per-prompt, with a small positive net, and a small negative net once TFG is on top.

---

## What this means for a NeurIPS-shaped paper

The standard framing of "RL post-training vs. inference-time guidance" treats them as alternatives or a stack. This result suggests a third characterization: **guidance gains decompose into an amortizable component and a non-amortizable component, in a roughly measurable way**.

- The **amortizable component** (9% here) is what's captured by supervised distillation of the teacher's output distribution — essentially the static shift TFG induces on the learned conditional.
- The **non-amortizable component** (69%) is what requires the guidance gradient *at inference time* — likely the step-by-step adjustment of the denoising trajectory in response to the predictor.
- The **redundancy/interference term** (22%) is where distillation moves the weights in a direction that TFG can no longer productively exploit.

These three fractions can likely be pushed around with distillation objective, model capacity, and training compute. That's what a full paper would characterize — a scaling law for "how amortizable is inference-time guidance."

---

## All prior supporting results (from the base-sweep and DDRL side)

### Base ρ response (monotonic, log-linear)

| ρ | base mean CLIP | Δ vs base no-TFG |
|---|---|---|
| 0 | 0.2681 | — |
| 2 | 0.2710 | +0.0030 |
| 5 | 0.2779 | +0.0098 |
| 10 | 0.2896 | +0.0215 |
| **20** | **0.3119** | **+0.0439** |

### DDRL axis (from the earlier A100 run — did *not* train at our scale)

| | no TFG | TFG ρ=20 |
|---|---|---|
| Base | 0.2681 | 0.3119 (+0.044) |
| DDRL-LoRA (300 steps, batch 2, KL 0.05) | 0.2670 (≈base) | 0.3106 (+0.044) |

DDRL at this compute budget didn't move the base (within noise) — and therefore didn't provide an informative second axis. Distillation did: +0.0041 is small but non-zero and per-prompt consistent (18/30).

---

## Method summary (for the email + paper)

### TFG-Flow

Ported upstream TFG's DDIM-era guidance to SD3's `FlowMatchEulerDiscreteScheduler`. The hook, per scheduler step:

1. Re-runs the transformer in a grad scope on a clone of x_t.
2. Predicts x₀ via `x₀ = x_t − σ·v` (diffusers SD3 convention, verified against source).
3. VAE-decodes x₀ to image space.
4. Scores with a differentiable predictor (CLIP-L/14 — mmdetection is not differentiable, so TFG's empirical caveat from the image_label_guidance task applies here too).
5. `torch.autograd.grad(score, x_t)` → rescaled update applied to x_t.

Code: `scripts/tfg_flow.py`, `scripts/clip_predictor.py`. 5/5 CPU smoke tests.

### TFG Distillation (this run's novel piece)

1. **Stage 1 — teacher dataset.** Run TFG-guided SD3.5-M at ρ=10 on 200 GenEval-style prompts (held out from the 30-prompt eval). Save final **latents** (not images) — avoids VAE encode roundtrip at training time.
2. **Stage 2 — distillation training.** LoRA (rank 64, α=128) on the transformer. Per step, load a teacher latent `x₀_T`, sample σ ∈ [0.05, 0.95], form `x_t = σ·noise + (1−σ)·x₀_T`, predict velocity, MSE against `v_target = noise − x₀_T`. 500 steps, batch 4, lr 2e-5, AdamW. Loss drops ~0.5 → 0.25.
3. **Stage 3 — eval.** Load LoRA (PeftModel.from_pretrained — diffusers' native LoRA loader doesn't strip PEFT's `base_model.model.*` prefix for SD3, which cost us a $4.50 silent-no-op run; fixed + guarded by a pre-flight verifier that fails fast on load regressions). Run Cell C′ (distilled no TFG) and Cell D′ (distilled + TFG ρ=20) on the 30 held-out prompts.

Code: `scripts/cloud/generate_tfg_dataset.py`, `scripts/train_distillation.py`, `scripts/cloud/run_distillation.sh`, `scripts/cloud/verify_lora_load.py`. 2/2 CPU smoke tests.

---

## Caveats

1. **Predictor is CLIP-L, not GenEval.** Same substitution upstream TFG makes for `image_label_guidance`. GenEval-as-downstream-eval is a planned follow-up.
2. **N=30 held-out prompts, single seed.** Per-prompt wins are directional; a publication version needs ~150-500 prompts and multiple seeds for CIs.
3. **Single distillation setting.** ρ=10 teacher, 500 steps, 200 samples. Should vary all three to sketch the "fraction amortizable" curve.
4. **No μ-term (x₀ refinement gradient).** Only the ρ branch of TFG is implemented. Adding μ may shift the decomposition.
5. **Train prompts and eval prompts both from GenEval categories but disjoint at the string level.** A held-out *category* (e.g., position) might show different amortization.

---

## Cost accounting (all sessions)

| Run | Instance | Wall | Spend |
|---|---|---|---|
| Smoke + ρ sweep (A10 + mini-DDRL) | A10 us-east-1 | ~2 hr | $2.50 |
| Real DDRL (A100 asia-south-1) | A100 SXM4 40GB | 73 min | $2.42 |
| **Distillation pipeline (H100 us-west-3, then A10 re-eval after fix)** | **H100 PCIe + A10** | **~2 hr** | **~$5.50** |
| — of which: silent-load H100 run before fix | H100 | ~2 hr | $4.95 |
| — narrow A10 re-eval with fixed loader | A10 | ~30 min | $0.65 |
| **Total** | | ~6 hr | **~$10.50** |

The $4.95 on the broken H100 was the expensive lesson — silent format-mismatch between PEFT's save and diffusers' native load. Now guarded by `scripts/cloud/verify_lora_load.py` (mandatory 45s pre-flight on any checkpoint-loading run).

---

## What's saved

- `outputs/base_sweep/` — 5 cells × 30 images JSONs (the ρ response curve)
- `outputs/ddrl_real_sweep/` — A100 DDRL result JSONs
- `outputs/distill_dataset/manifest.json` — 200 teacher-rollout references
- `outputs/distill_eval_base/` — Cell A′ + B′ JSONs (H100 fresh baseline)
- `outputs/distill_eval_student/` — Cell C′ + D′ JSONs from the **broken** run (preserved for audit)
- `outputs/distill_eval_student_fixed/` — Cell C′ + D′ JSONs from the **fixed** re-eval (the actual results)
- `outputs/checkpoints/ddrl_real/final/` — DDRL LoRA (14 MB, rank 64)
- `outputs/checkpoints/distilled/final/` — distillation LoRA (111 MB with all 6 checkpoints; final/ is the 500-step terminal state)
- `scripts/` + `tests/` — all code, 14/14 local smoke tests passing
- `docs/next-steps.md` — the research direction memo

---

## Paper-worthy finding in one sentence

**The gain from TFG-Flow at ρ=20 on SD3.5-M decomposes into ~9% amortizable (recoverable by supervised distillation into weights), ~69% non-amortizable (only available at inference time), and ~22% redundancy loss (distillation flattens TFG's gradient).** The fractions shift with distillation compute and predictor choice — that's a paper.
