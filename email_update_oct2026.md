Subject: Re: Weights or Sampling

Hi Haotian,

I never got an answer on framing vs structural, so I went with structural. Two things from October.

The conditions map now predicts out of sample, with error bars. Three new rewards (HPSv2, SigLIP, CLIP on Pick-a-Pic), predictions written down before training, GRPO at three seeds each. Steepness got the guidance ordering right on all three, eight for eight. The cosine coordinate failed and is dropped. With seeds, RL's gain on SigLIP and CLIP/Pick-a-Pic is zero within noise while guidance stays positive. Guidance matches or beats tuned GRPO on every differentiable reward we have; RL only draws level on aesthetic and PickScore.

Second: DRaFT trains the weights on the gradient guidance uses. At lr 1e-4 it matches guidance's best reward for an hour, then collapses into CLIP-adversarial texture, three seeds of three. At lr 3e-5 it never collapses and reaches +0.136 CLIP in 6 hours, above guidance's +0.115, but the same texture creeps in from the first hour (PickScore -0.044 vs guidance's -0.003; figure attached). So the learning rate sets the speed of the hacking, not whether it happens. Guidance gets the reward without it. Fair distillation finishes today.

ICML in January: a human audit and a pi0.5 LIBERO test on the released piRL checkpoints.

20 minutes next week? Main question: is the DRaFT contrast the main figure or a side result.

Best,
Nihal

---
Attachments: docs/figs/draft_lr3e-5_base_vs_guidance_vs_6h.png and the table below (GRPO column is the mean over seeds; ± is the SD across seeds where there are three).

| setting | gradient cosine | slope (norm.) | guidance @ρ20 | GRPO β=0 | RL share |
|---|---|---|---|---|---|
| aesthetic | 1.00 | 0.174 | +0.140 | +0.142 (3 seeds) | 1.02 |
| ensemble | 0.66 | 0.019 | +0.032 | +0.016 | 0.51 |
| CLIP / GenEval | 0.64 | 0.041 | +0.115 | +0.048 (2 seeds) | 0.42 |
| SigLIP | 0.64 | 0.032 | +0.021 | +0.001 ± 0.003 | 0.07 |
| PickScore | 0.58 | 0.006 | +0.007 | +0.007 | 0.98 |
| HPSv2 | 0.49 | 0.020 | +0.014 | +0.011 ± 0.007 | 0.74 |
| CLIP / OCR | 0.44 | 0.061 | +0.173 | +0.026 | 0.15 |
| CLIP / Pick-a-Pic | 0.33 | 0.074 | +0.204 | +0.001 ± 0.010 | 0.00 |
