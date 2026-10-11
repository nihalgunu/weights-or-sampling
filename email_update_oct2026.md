Subject: Weights or Sampling — two updates

Hi Haotian,

Two results since the workshop version, on your August question of when training helps.

1. Added the missing arm: DRaFT, training weights on the reward gradient guidance uses. On CLIP it reaches +0.13 in an hour (guidance +0.115, GRPO +0.048), then collapses into CLIP-adversarial texture, in both runs. A velocity-KL term stops the collapse but not the trade: a +0.10 plateau while PickScore and compressibility keep degrading. Guidance matches it with flat cross-metrics. Images attached.

2. Pre-flight diagnostic on three new rewards (HPSv2, SigLIP, CLIP/Pick-a-Pic), predictions written down first. Steepness orders the guidance gain on all eight settings. Prompt-dependence only gets the extremes right, so it stays an ordering, not a predictor. Table attached.

Next: seeds on the new settings and DRaFT on the other rewards, aiming at ICML. Question: is the hacking contrast main-text material or a side result to you?

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
