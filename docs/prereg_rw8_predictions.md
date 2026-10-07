# Pre-registered predictions for the three new reward settings (written 2026-10-05 10:45, before any RL result exists)

Basis: the five measured settings (CLIP/GenEval, CLIP/OCR, PickScore, aesthetic, ensemble)
give the ordering "cross-prompt gradient cosine orders RL's amortizable share of the guidance
gain" (aesthetic 1.00 -> share ~1.0; CLIP 0.64 -> 0.42; PickScore 0.58 -> 0.07*; OCR 0.44 -> 0.15).
(*PickScore RL is at beta=0.04, not the tuned regime.)

Measured pre-flight cosines for the new settings (study_pdep_rw8.json, 8 images x 16 prompts):

| setting                    | cosine |
|---------------------------|--------|
| CLIP on Pick-a-Pic prompts | 0.33   |
| HPSv2 on GenEval prompts   | 0.49   |
| SigLIP on GenEval prompts  | 0.64   |

Predictions (RL = FlowGRPO group 8, beta 0, 4h on one H100, seed 42; guidance = TFG rho 20):

1. RL's share of the guidance gain orders SigLIP > HPSv2 > CLIP/Pick-a-Pic.
2. CLIP/Pick-a-Pic has the lowest share of any setting measured so far (below OCR's 0.15).
3. SigLIP's share lands near CLIP/GenEval's (~0.4), since their cosines match (0.64).
4. Guidance beats RL on all three (ratio > 1), consistent with every differentiable setting so far.

Coordinate 1 (dose-response slope) is measured by the same run; the prediction on it is only
that the slope orders the guidance gain at rho 20 across the three, as it did for the first four.

Falsification: if the observed shares do not follow ordering (1), the paper keeps calling the
diagnostic an ordering anchored by extremes, not a predictor. A single-seed RL run is noisy
(seed spread up to +/-0.02 on CLIP), so (1) is judged on the mean if more seeds are added later.

## Outcome (scored 2026-10-06, RL = FlowGRPO β=0 seed 42, 4h H100; guidance = TFG ρ20, 150 prompts × 1 seed)

| setting | cosine | guidance Δ@ρ20 | RL Δ | RL share | predicted rank | observed rank |
|---|---|---|---|---|---|---|
| SigLIP / GenEval | 0.64 | +0.021 | +0.004 (t 2.4) | 0.19 | 1 (highest) | 2 |
| HPSv2 / GenEval | 0.49 | +0.014 | +0.018 (t 9.6) | 1.26 | 2 | 1 |
| CLIP / Pick-a-Pic | 0.33 | +0.204 | +0.009 (t 1.1) | 0.04 | 3 (lowest) | 3 |

1. Ordering SigLIP > HPSv2 > Pick-a-Pic: **partly wrong** (HPSv2 and SigLIP swapped; Pick-a-Pic lowest as predicted).
2. Pick-a-Pic lowest share of any setting, below OCR's 0.15: **held** (0.04).
3. SigLIP share near CLIP/GenEval's ~0.4: **failed** (0.19).
4. Guidance beats RL on all three at matched strength: **held** (at ρ=20 for SigLIP and Pick-a-Pic; HPSv2 needed ρ≥80, where guidance reaches +0.077 vs RL +0.018).

Over all 8 settings: slope (fit on ρ≤10) vs normalized guidance gain at ρ=20, Spearman 1.0;
cosine vs normalized RL gain, Spearman 0.60. The steepness coordinate is a predictor; the
prompt-dependence coordinate orders the extremes (cos 1.00 → share ~1; cos 0.33 → 0.04) but not
the middle, where single-seed RL noise (±0.02 on CLIP) is larger than the differences.
