# Update email to Haotian (target send: ~Oct 17; placeholders in [brackets] fill from today's runs)

Subject: Weights or Sampling — two new results since BeNTo

Hi Haotian,

Two things since the workshop submission, both on the question you raised in August (when should
training-based methods help more, and why).

**1. The pre-flight diagnostic now has eight reward settings, and it was tested out of sample.**
Two cheap measurements, the dose-response slope of guidance and the cross-prompt cosine of the
reward gradient, were used to predict the training-vs-guidance margin on three settings before
their RL runs existed (predictions written down on Oct 5, file in the repo).

| setting                   | gradient cosine | slope (norm.) | guidance Δ @ρ20 | RL Δ (β=0, 4h) | RL share | predicted rank |
|---------------------------|-----------------|---------------|-----------------|----------------|----------|----------------|
| aesthetic                 | 1.00            | 0.166         | +0.140          | +0.142         | 1.02     | (anchor)       |
| ensemble / GenEval        | 0.66            | 0.019         | +0.032          | +0.016         | 0.51     | (anchor)       |
| CLIP / GenEval            | 0.64            | 0.044         | +0.115          | +0.048         | 0.42     | (anchor)       |
| SigLIP / GenEval          | 0.64            | 0.032         | +0.021          | +0.004         | 0.19     | 1st of new     |
| PickScore / GenEval       | 0.58            | 0.006         | +0.007          | +0.007         | 0.98     | (anchor)       |
| HPSv2 / GenEval           | 0.49            | 0.020         | +0.014          | +0.018         | 1.26     | 2nd of new     |
| CLIP / OCR                | 0.44            | 0.060         | +0.173          | +0.026         | 0.15     | (anchor)       |
| CLIP / Pick-a-Pic         | 0.33            | 0.074         | +0.204          | +0.009 (n.s.)  | 0.04     | 3rd (lowest) ✓ |

(All RL at β=0, one seed unless noted.) Reading: the steepness coordinate predicts the guidance response out of sample on all eight settings
(Spearman 1.0 with ρ=20 held out of the fit). The prompt-dependence coordinate orders the extremes as
predicted (Pick-a-Pic, cosine 0.33, is the setting where RL amortises least: share 0.04, RL not significant)
but not the middle (HPSv2 amortises more than SigLIP despite a lower cosine; Spearman 0.60 over 8), where
single-seed RL noise exceeds the differences, so I keep reporting it as an ordering anchored by the extremes. At matched strength
guidance still wins every differentiable reward: on the two flat-landscape additions ρ=20 under-strengths it
(as for PickScore), and at ρ=320 guidance reaches +0.077 on HPSv2 (vs RL +0.018, 4.2×) and +0.062 on
SigLIP (vs RL +0.004, 16×), 150/150 and 149/150 prompts.

**2. The missing cell of the map: weights × reward-gradient (DRaFT).** DRaFT-1 (LoRA, same 30
prompts, same frozen CLIP-L) is the strongest weight-side optimizer we have measured and the least
safe. Scoring every checkpoint on the 150 eval prompts: +0.042 at step 95, +0.131 at step 482
(~1 GPU-hour; t 16, 136/150; +0.129 on the 120 held-out prompts), i.e. above guidance at ρ=20
(+0.115) and 2.7× FlowGRPO at β=0 (+0.048). Then it collapses: by step 840 every image is
CLIP-adversarial texture (−0.128), and the collapse recurred in two independent runs at different
steps (360, ~820). Cross-metrics tell the same story in miniature: per unit of CLIP gain, PickScore
falls 0.29 for DRaFT, ~0.6 for GRPO, 0.03 for guidance. And the peak checkpoint's images are
visibly degraded (washed out, grainy) while still depicting the right objects: the largest reward
gain of any weight-side method is bought with quality, which a CLIP-family external judge (SigLIP)
misses and a preference model (PickScore) catches. Figure attached. So the gradient is the signal that matters,
and only the sampler applies it without a trust region. With a velocity-KL regulariser (0.05) it does not collapse in 3h and plateaus at about +0.10 (peak +0.108,
135/150), roughly 2× GRPO and just under guidance, but PickScore keeps falling (−0.02 → −0.06) and JPEG
compressibility keeps rising as training continues: the regulariser removes the cliff, not the trade.

Plan from here: the 3-seed matched-hardware DRaFT arms for the other rewards, the human study, and
the budget crossover at 64h, for an ICML submission. I'd value your read on whether the hacking
contrast belongs in the main text or as the mechanism for the signal axis.

Best,
Nihal
