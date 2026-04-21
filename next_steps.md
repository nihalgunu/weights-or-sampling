# Next experiments to run — post-pilot

Current state: ρ curve + single-point decomposition + training-compute scaling
(Phase A) done, committed, pushed to main. Below: every remaining experiment,
ranked by impact × feasibility. Compute estimates assume A100 SXM4 at
$1.99/hr unless noted.

---

## Tier 1 — required to be NeurIPS-complete

### 1. Phase B: teacher-ρ scaling sweep (**highest ROI**)

**What:** 3 new full distillations at teacher ρ ∈ {2, 5, 20}, each being
Stage 1 (200 teacher latents) + Stage 2 (500-step LoRA) + Stage 3 (Cell C
and Cell D at eval ρ=20). Combined with the existing ρ=10 run, gives a
4-point scaling curve on teacher strength.

**Why:** The #1 question a reviewer asks is "how does your decomposition
change with teacher strength?" Without this, Claim 4 of the paper outline is
missing. With it, we have a 2D scaling (training-compute × teacher-strength)
instead of a single axis.

**Cost / time:** ~$9 / ~4.5 hr on A100. Fully automated — launch, walk away.

**How to run:**
```bash
# on any Lambda instance after SD3.5-M cached (remote_setup.sh)
bash scripts/cloud/run_teacher_rho_sweep.sh
```
Outputs: `outputs/distill_dataset_rho{2,5,20}/`,
`outputs/checkpoints/distilled_rho{2,5,20}/`, `outputs/distill_eval_rho{2,5,20}/`.

**Expected finding (hypothesis):** amortizable fraction grows with teacher ρ
(stronger teacher → more to distill), but residual at inference might shrink
(less room for TFG to re-steer). If that holds, there's an *optimal teacher
strength* for the stacked pipeline.

### 2. GenEval evaluation on all generated images

**What:** Install mmdetection + GenEval detector models, run
`repos/geneval/evaluation/evaluate_images.py` on every image directory we've
already produced (cell_a/, cell_b_rho*/, cell_c/, cell_d/ across both base
and distilled). Re-compute Cell-level numbers using the **actual** metric
Haotian's paper uses.

**Why:** The CLIP-predictor caveat is the first thing any TFG/SD3 reviewer
will flag. Reconfirming the decomposition on GenEval removes the caveat and
makes the claim canonical. Images already exist; this is eval-only.

**Cost / time:** ~$2–4 on A10. Biggest risk is the mmdetection install
(pinned torch/mmcv/mmdet matrix is notoriously flaky) — time-box to 30 min;
if stuck, the fallback is a simpler OWLv2 / Grounding DINO object-check.

**How to run:**
```bash
bash scripts/cloud/install_geneval.sh   # already written, time-boxed to 30 min
python3 scripts/cloud/run_geneval_eval.py --image_dir outputs/.../cell_X  # to write
```
(The second script is NOT yet written — ~50 LOC, loops over cell dirs and
invokes GenEval's `evaluate_images.py` per cell. Low risk code.)

### 3. Multi-seed replication (2 additional seeds)

**What:** Re-run Cells A, B, C, D on the same 30 held-out prompts with seeds
`100 + prompt_id` and `200 + prompt_id` (in addition to the existing
`42 + prompt_id`). Gives 3-seed CIs for every reported number.

**Why:** Current replication variance is one accidental data point
(ckpt-500 run twice — 0.2725 vs 0.2713, spread 0.0012). Proper CIs need
≥3 seeds. Reviewer asks "is +0.0041 significantly > 0?" — we currently
cannot answer rigorously.

**Cost / time:** ~$4–6 / ~2–3 hr on A100. Same 30 prompts so each run
is ~30 min × 2 seeds = ~1 hr for the base + TFG axis; plus another hour
for distilled + distilled-TFG.

**How to run:** small script that loops over seeds and calls the existing
`run_rho_sweep.py` with `--seed`. (`run_rho_sweep.py` already has `--seed`.)

---

## Tier 2 — strengthens the paper without being required

### 4. Larger eval prompt set (N → ~150–500)

**What:** Scale the eval set from 30 prompts to the full GenEval eval set
(553 prompts) or a significant subset (~150). Re-run Cells A, B, C, D at
this scale.

**Why:** N=30 gives wide CIs. N=150 would halve them. N=553 is the
canonical scale.

**Cost / time:** ~$5–15 / ~3–8 hr on A100, depending on N. Linear in N.

**Implementation note:** just change the `--prompts_file` to
`repos/geneval/prompts/evaluation_metadata.jsonl` after cloning GenEval.

### 5. Implement the μ-term (x₀-refinement gradient)

**What:** Upstream TFG has two branches: ρ (gradient on x_t, which we have)
and μ (iterative refinement on the x₀ estimate, which we don't). Adding μ
may shift the decomposition — if μ amortizes differently from ρ, that's a
separate result.

**Why:** Completes the TFG method definition; answers "are you sure you
tried real TFG?" Around 40 LOC change in `scripts/tfg_flow.py` plus new
CLI arg + tests.

**Cost / time:** ~1 day of coding + local CPU tests + $3–5 to re-measure
the base Cells B and D at matched ρ/μ.

### 6. Cross-category generalization test

**What:** Train the distillation on prompts from GenEval categories
{single_object, two_objects, counting} (the 200 we used) and evaluate
*only* on {position, colors, color_attribution} — 3 held-out categories
our current eval set already mixes in. Measures whether distillation
transfers *across category*, not just across prompt string.

**Why:** Stronger generalization claim than "held out at string level."

**Cost / time:** Zero new training — just re-score existing Cell C and D
outputs restricted to the 3 held-out categories (there are 18 such prompts
in our eval set of 30). Analysis-only change.

---

## Tier 3 — extends scope beyond current paper

### 7. Swap predictor (SigLIP, aesthetic score)

**What:** Re-run Cells B/D with a different differentiable predictor. Does
the amortization fraction depend on *what* we guide on?

**Why:** Tests whether the 8% figure is about TFG-the-method or
CLIP-the-predictor. If swapping predictors changes the number significantly,
the decomposition framework is still valid but the specific number is
predictor-dependent.

**Cost:** ~$5–10 per new predictor. 1 day to write (one predictor class
per alternative).

### 8. Port to different base model

**What:** Same pipeline on SDXL (UNet) or Flux (flow-matching like SD3).

**Why:** Tests whether the decomposition is SD3-specific or general. Flux
would be the more compelling comparison (same flow-matching family); SDXL
would test generality to UNet-style diffusion.

**Cost / time:** ~1–2 weeks for the port (TFG hook needs adapting to the
different pipeline APIs; LoRA targets differ); compute per model ~$15–30.

### 9. Video TFG-Flow (the original pitch)

**What:** Extend TFG-Flow to a video flow-matching model (CogVideoX, Wan,
HunyuanVideo). This was Nihal's original pitch to Haotian.

**Why:** High-ceiling / high-variance. First TFG-for-video-flow
demonstration is a landmark if it works.

**Cost / time:** ~$100–500, weeks. Temporal-coherence risk is high.

---

## Recommended next action

**Run Phase B today** ($9, 4.5 hr). That gets Claim 4 of the paper, makes
the scaling story 2D. Everything else can wait.

After Phase B:
- **If teacher-ρ sweep shows a clean monotonic trend** → paper has its
  scaling law. Next priority: GenEval eval (remove the CLIP-proxy caveat).
- **If the trend is noisy / non-monotonic** → multi-seed replication first,
  before claiming anything about teacher-strength scaling.

---

## Already-committed code / data that Phase B and beyond will reuse

- `scripts/cloud/run_teacher_rho_sweep.sh` — Phase B orchestrator, verified
- `scripts/cloud/run_full_scaling.sh` — Phase A+B master orchestrator
- `scripts/cloud/verify_lora_load.py` — mandatory pre-flight on any
  distilled checkpoint load; 45s of GPU
- `scripts/cloud/prompts_train.json` — 200 distillation-train prompts
  (held out from our 30-prompt eval)
- `scripts/cloud/prompts_smoke.json` — 30 held-out eval prompts
- `outputs/distill_eval_base/` — Cell A′, Cell B′ baseline (do not re-run)
- `outputs/ckpt_sweep/` — Phase A checkpoints 100-500 (do not re-run)
- `outputs/checkpoints/distilled/{checkpoint-{100,200,300,400},final}/` —
  the 5 training-compute checkpoints from the original distillation
- 14/14 local CPU smoke tests (tfg_flow: 5, train_ddrl: 7, distillation: 2)

**Rotate the Lambda API key and HF token before any further runs.** Both
have been in chat history many times. Links:
- https://cloud.lambdalabs.com/api-keys
- https://huggingface.co/settings/tokens
