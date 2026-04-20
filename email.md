Hi Haotian,

How it went — I ran TFG-Flow on SD3.5-M and then asked a follow-up: how much of TFG's gain is amortizable into model weights?

**Setup.** 30 held-out GenEval-style prompts, 512², 28 steps, CLIP-L predictor. TFG-Flow at ρ=20 gives Δ=+0.045 CLIP on base (30/30 per-prompt wins, clean monotonic response in ρ∈{2,5,10,20}).

**Distillation experiment.** 200-prompt teacher pool (disjoint from eval), TFG-guided rollouts at ρ=10, saved latents. LoRA r=64 finetune with supervised flow-matching loss against teacher latents. Then evaluated on the 30 held-out prompts:

|              | no TFG | TFG ρ=20 |
|--------------|--------|----------|
| Base         | 0.2684 | 0.3132   |
| Distilled    | 0.2725 | 0.3035   |

**Decomposition.** Of TFG's +0.045 base effect:
- **9% amortized** into weights (C'−A' = +0.0041, 18/30 wins)
- **69% residual** at inference after distillation (D'−C' = +0.0310, 29/30 wins)
- **22% lost** to redundancy (distillation shrinks TFG headroom)

Stacked gain (D') is strictly below TFG-alone (B') at this setting — the combined pipeline is partially redundant, not complementary. The fractions should move with distillation compute and predictor; sweeping that is the paper.

Code is on SD3.5-M flow matching; CPU smoke tests 14/14 pass; all cloud runs under $15 total. Thinking this framing — "how amortizable is inference-time guidance?" — is worth a NeurIPS writeup since it decomposes your TFG gain into a concrete and scalable-law-shaped question. Happy to share more detail.

Best,
Nihal
