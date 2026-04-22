Hi Haotian,

Quick update on TFG-Flow on SD3.5-M. On 30 held-out GenEval-style prompts, ρ=20 gives Δ=+0.045 CLIP over base (30/30 per-prompt wins, clean monotonic response in ρ).

I distilled TFG teacher trajectories into a LoRA and decomposed the gain across 5 training-compute points: the **amortizable fraction stays in a narrow 3–10% band**, **inference-time residual stays at 70–77%**, and more training *slightly hurts* the stacked pipeline. Distillation saturates within ~100 steps — more training compute does not extract more of the teacher. That stability is the unexpected part.

Teacher-ρ sweep is the natural next axis (already scripted, ~$9 / 4.5 hr to run). Feels like a NeurIPS-shape question: "How amortizable is inference-time guidance?"

Best,
Nihal
