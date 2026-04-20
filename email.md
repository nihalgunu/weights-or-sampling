Hi Haotian,

How it went — TFG-Flow is up on SD3.5-M. Clean monotonic CLIP response on 30 GenEval-style prompts at 512²: Δ = +0.003 / +0.010 / +0.022 / +0.044 at ρ = 2 / 5 / 10 / 20. Bonus finding: TFG's Δ at ρ=20 was identical (+0.044) on the base model and on a mini-DDRL-tuned LoRA variant — small weight perturbations don't shift TFG's gain, a clean invariance property.

Mini-DDRL itself was too noisy at my compute budget to move the base, so RL-vs-TFG complementarity is still open on that axis. Pivoting next to TFG-trajectory distillation: supervised flow-matching loss on (prompt, TFG-image) pairs. Low-variance signal where CLIP-reward RL was too noisy, and it cleanly tests whether TFG has residual gain after the weights absorb most of it.

Will share when I have numbers.

Best,
Nihal
