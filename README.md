# Weights or Sampling? A Controlled Study of Reward Optimization in Flow-Matching Models

Research code and data for the controlled comparison of training-free guidance
(TFG) vs. GRPO-style RL post-training (FlowGRPO) on SD3.5-Medium, per
individual reward model, at matched single-GPU training budget.
Nihal Gunukula (Purdue) and Haotian Ye (Stanford).

**Paper:** preprint coming soon. This repository releases the code and the
per-prompt evaluation matrices; every number in the paper recomputes from the
matrices in `outputs/` under the prompt-clustered recipe described below.

## Headline results (publication-grade run, Aug 2026)

150 prompts x 3 eval seeds per arm, prompt-clustered statistics (n=150),
3 independent RL training runs per reward, external metrics included.

| | CLIP-L / GenEval-style | CLIP-L / OCR | PickScore |
|---|---|---|---|
| TFG (headline rho) | +0.1152 (150/150 prompts) | +0.1733 (150/150) | +0.0329 (150/150) |
| RL, 3 training runs | +0.016 .. +0.023 | +0.003 .. +0.011 (mostly n.s.) | -0.012 .. +0.008 |
| TFG minus pooled RL | t = +16.0 | t = +27.6 | t = +22.7 |

- TFG beats matched-budget FlowGRPO on all three rewards; small-scale RL is
  high-variance across training runs (one PickScore run actively regressed).
- Stacking TFG on an RL checkpoint beats TFG-alone in 7 of 9 cells and never
  falls below the checkpoint alone.
- External-metric caveat: on OCR prompts TFG's CLIP-L gain slightly degrades
  real text readability (EasyOCR) while RL slightly improves it. Reward is
  not the task; stronger optimizers inherit the reward's blind spots harder.
- Detector-based scoring (GenEval-style, OWLv2 + box logic + CLIP color
  crops): object-level correctness sits near ceiling (~0.90) and no arm moves
  it significantly; the failed PickScore RL run is independently confirmed as
  a genuine regression (detector -0.027, t -2.4).

## Reproduction

Every run in the paper is reproducible from the released scripts: training
uses fixed initialization and rollout seeds, and every reported number
recomputes from the released per-prompt evaluation matrices under the
prompt-clustered recipe in the paper (Sec. 3). The tracked release is the
evaluation matrices and the seeded scripts; trained LoRA adapters for retained
runs are available from the authors on request (they are not committed, and raw
training logs are not part of the release). All code pins
`transformers==4.49.0` — newer releases break the SD3 pipeline import.

Data (all numbers in the paper recompute from these):

- Matrices: `outputs/study_{clip,clip_ocr,pickscore}_pub_matrix.json`
  (per-prompt scores, all metrics, all seeds); every other paper experiment
  has its own `outputs/study_*_matrix.json` (see "Paper mapping" below)
- External metrics: `outputs/study_{clip,clip_ocr,pickscore}_pub_external.json`
- Detector scoring: `outputs/study_{clip,pickscore}_pub_detector.json`
  (`scripts/cloud/detector_score.py`; CLIP-setting arms regenerated via
  `scripts/cloud/run_detector_eval.sh`)
- LoRA adapters: retained locally for `flowgrpo_*_pub_s{42,1042,2042}`, not
  committed (available on request); all runs reproduce from the seeded runners
  regardless
- Image archives: `outputs/study_{clip_ocr,pickscore}_pub_images.tar.gz`
  (for GenEval / human eval)

Paper mapping (paper section -> matrix -> runner):

- beta=0 fair baseline (4.2): `study_beta0_{clip,16h,16h_s1042,ocr,ens}_matrix.json`
  (`run_beta0.sh`, `run_beta0x.sh`)
- Budget crossover (4.2): `study_xover{8,16,32}h_matrix.json` (`run_xover.sh`)
- KL ablation (4.2): `study_kl_matrix.json` (`run_kl.sh`; the beta=0.04 arm
  reuses the pub-matrix s2042 checkpoint, disclosed in the paper)
- Group-size threshold (4.2): `study_group_matrix.json` (`run_group.sh`)
- Dose-response sweeps (4.3): `study_{clip,clip_ocr,pickscore}_matrix.json`,
  `study_pickscore_rho{ext,ext80,ceil}_matrix.json`, `study_highrho_matrix.json`
- Aesthetic reward (4.3): `study_rw6_aesthetic_matrix.json` (`run_rw6.sh`),
  seed replication `study_aes_seeds_matrix.json` (`run_aes_seeds.sh`)
- Prompt-dependence coordinate (4.3): `study_pdep.json` (`run_pdep.sh`)
- mu branch / composition (4.4): `study_mu_matrix.json` (`run_mu.sh`),
  `study_ensemble_matrix.json` (`run_ens.sh`), `study_pack_*_matrix.json`
  (`run_pack.sh`)
- GenEval-553 (4.4): `study_g553_{matrix,detector}.json` (`run_g553.sh`)
- SD3.5-Large replication (3): `study_large_tfg_matrix.json` (`run_large_tfg.sh`)
- Best-of-N (4.4): `study_bon{,16}_clip_matrix.json`,
  `study_bon_pickscore_matrix.json` (`run_bon.sh`)
- OCR conversion ladder (4.5): `study_ladder_{matrix,taskocr}.json`
  (`run_ladder.sh`)
- Compositional hard set (4.5): `study_comphard_{matrix,detector}.json`
- Cross-domain transfer (4.1): `study_shift_{clip2ocr,ocr2gen}_matrix.json`
- BLIP-ITM judge (4.5): `study_imagereward_external.json` (`run_ir_only.sh`;
  despite the filename it holds BLIP-ITM `Salesforce/blip-itm-large-coco`
  scores, not ImageReward)
- JPEG compressibility RL (Table 2): `flowgrpo_eval.json`
  (`eval_lora_compressibility.py`; early 30-prompt protocol)

Analysis:

```bash
python3 scripts/analyze_pub.py        # publication-grade tables
python3 scripts/analyze_study.py      # exploratory (n=90) tables
```

Harness (Lambda Cloud, 1x H100-80GB per reward):

```bash
# end-to-end: 3 training seeds + 150-prompt matrix + external metrics
bash scripts/cloud/babysit_pub.sh <clip|clip_ocr|pickscore> 4
```

Key components:

- `scripts/cloud/run_pub_reward.sh` - on-box orchestrator (train x3 + eval)
- `scripts/cloud/eval_study_matrix.py` - multi-arm eval (base, TFG rho-sweep,
  RL per seed, stacked), paired seeds, cross-metric scoring
- `scripts/pickscore_predictor.py` - differentiable PickScore for TFG guidance
- `scripts/cloud/external_metric.py` - SigLIP + EasyOCR scoring
- `scripts/tfg_flow.py` - TFG rho-branch guidance hook for SD3 flow matching
- `scripts/cloud/run_flowgrpo_*.sh` - FlowGRPO training runners (FG_GROUP,
  FG_NBPE, FG_SEED, FG_BETA env knobs; FG_BETA overrides the policy-KL
  coefficient, upstream default 0.04 - the paper's fair-baseline RL runs set
  FG_BETA=0, see run_beta0.sh / run_kl.sh)

Requires `.env` with `LAMBDA_API_KEY` and `HF_TOKEN` (SD3.5-M is gated), and
an SSH key registered with Lambda (`~/.ssh/lambda_claude`).

## Repo history

Earlier phases (visible in git history and `outputs/`): TFG rho-sweep and
distillation decomposition (April; `outputs/{base,ckpt,ddrl}_sweep/`,
`outputs/distill_*/`), FlowGRPO group-size investigation (May-June, group=2
fails / group=8 banks reward), exploratory TFG-vs-GRPO matrix and rho
calibration (August), publication-grade run and conditions-map wave
(mid-late August).

## License

Apache License 2.0 (see `LICENSE`). SD3.5 model weights are governed by the
Stability AI Community License; reward models by their respective licenses.
