Hi Haotian,

Quick update on TFG-Flow on SD3.5-M. On 30 held-out GenEval-style prompts, ρ=20 gives Δ=+0.045 CLIP over base (30/30 per-prompt wins, clean monotonic response in ρ).

I then distilled a 200-prompt TFG teacher into a LoRA and decomposed the result: **~9% of TFG's gain is amortizable into weights, ~69% remains as inference-time residual, ~22% lost to redundancy**. Combined (distilled + TFG) is slightly below TFG-alone — partially redundant, not complementary.

Feels like a NeurIPS-shape question: "how amortizable is inference-time guidance?" Happy to share details or sketch the paper.

Best,
Nihal
