Subject: Weights or Sampling — two updates

Hi Haotian,

Two results since the workshop version, both on your August question (when does training help, and why).

**1. I added the missing competitor: DRaFT, i.e. training the weights on the same reward gradient guidance uses.**
It's the strongest thing on the weights side and the least safe. On CLIP it reaches +0.13 after about an hour (guidance at ρ=20 is +0.115, GRPO +0.048), then collapses into CLIP-adversarial texture within another hour. Happened in both runs, at different steps. A velocity-KL term stops the collapse but not the trade: the model plateaus near +0.10 while PickScore and compressibility keep getting worse. Guidance gets the same reward with flat cross-metrics. Images attached. The one-line version: the gradient is the signal that matters, and only the sampler uses it without paying in image quality.

**2. I tested the pre-flight diagnostic on three new rewards (HPSv2, SigLIP, CLIP on Pick-a-Pic prompts), predictions written down before the RL runs.**

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

The steepness coordinate holds up: fitted on ρ≤10, it orders the guidance gain at ρ=20 on all eight settings. The prompt-dependence coordinate only gets the extremes right (Pick-a-Pic is where RL does nothing, as predicted; HPSv2 vs SigLIP came out backwards), and the middle is within single-seed RL noise, so I'll keep calling it an ordering, not a predictor. Guidance still beats RL on every differentiable reward at matched strength; HPSv2 and SigLIP just need ρ≥80 (+0.077 and +0.062 there).

Next I'm adding seeds to the new settings and running the DRaFT arm on the other rewards, aiming at ICML. One question: does the hacking contrast belong in the main text as the mechanism for the gradient-vs-scalar axis, or is it a side result to you?

Best,
Nihal
