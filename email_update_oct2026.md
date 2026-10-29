Subject: Re: Weights or Sampling

Hi Haotian,

I never got an answer on framing vs structural, so I went with structural. Two things came out of October.

The conditions map now predicts out of sample. I added three rewards (HPSv2, SigLIP, CLIP on Pick-a-Pic) with predictions written down before training. Steepness got the guidance ordering right on all three, so it is eight for eight. The cosine coordinate failed, and I am dropping it to an ordering. Guidance at best strength beats tuned GRPO on every differentiable reward we have.

The more interesting one: I added DRaFT, which trains the weights on the same gradient guidance uses. For about an hour it matches guidance's best reward (+0.12 vs +0.115 CLIP), then it collapses into CLIP-adversarial texture. Three seeds, three collapses, between 45 minutes and 3 hours in. A KL term holds two of the three seeds at about +0.10 with visibly degrading images; the third collapses anyway. Guidance reaches the same reward cleanly. Same gradient, two places, very different safety. Images attached, you can see it.

Aiming for ICML in January: seeds, a human audit, and a pi0.5 LIBERO test using the released piRL checkpoints.

Do you have 20 minutes next week? Main thing I want your read on is whether the DRaFT contrast is the main figure or a side result.

Best,
Nihal

---
Attachments: docs/figs/draft_lr1e-4_base_vs_step482_vs_step740.png, docs/figs/draft_kl05_base_vs_step735_vs_final.png, and the table below.

| setting | gradient cosine | slope (norm.) | guidance @ρ20 | GRPO β=0 | RL share |
|---|---|---|---|---|---|
| aesthetic | 1.00 | 0.174 | +0.140 | +0.142 | 1.02 |
| ensemble | 0.66 | 0.019 | +0.032 | +0.016 | 0.51 |
| CLIP / GenEval | 0.64 | 0.041 | +0.115 | +0.048 | 0.42 |
| SigLIP | 0.64 | 0.032 | +0.021 | +0.004 | 0.19 |
| PickScore | 0.58 | 0.006 | +0.007 | +0.007 | 0.98 |
| HPSv2 | 0.49 | 0.020 | +0.014 | +0.018 | 1.26 |
| CLIP / OCR | 0.44 | 0.061 | +0.173 | +0.026 | 0.15 |
| CLIP / Pick-a-Pic | 0.33 | 0.074 | +0.204 | +0.009 (n.s.) | 0.04 |
