Subject: Re: Weights or Sampling

Hi Haotian,

I never got an answer on framing vs structural, so I went with structural. Two things came out of October.

The conditions map now predicts out of sample and has error bars. I added three rewards (HPSv2, SigLIP, CLIP on Pick-a-Pic) with predictions written down before training, then ran GRPO at three seeds on each. Steepness got the guidance ordering right on all three, so it is eight for eight. The cosine coordinate failed and I am dropping it. With seeds, the RL gain on SigLIP and CLIP/Pick-a-Pic is zero within noise while guidance stays clearly positive, so the split is cleaner than it looked with one seed: guidance matches or beats tuned GRPO on every differentiable reward we have, and RL only draws level on aesthetic and PickScore.

Second: DRaFT, which trains the weights on the same gradient guidance uses. At lr 1e-4 it matches guidance's best reward for about an hour, then collapses into CLIP-adversarial texture, three seeds out of three. A lower learning rate and a fair distillation baseline are running now; I will send those with images when they finish this week.

Aiming for ICML in January: a human audit and a pi0.5 LIBERO test using the released piRL checkpoints.

Do you have 20 minutes next week? Main thing I want your read on is whether the DRaFT contrast is the main figure or a side result.

Best,
Nihal

---
Attachment: the table below (GRPO column is the mean over seeds; ± is the SD across seeds where there are three).

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
