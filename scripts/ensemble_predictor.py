"""
Ensemble TFG predictor: weighted sum of CLIP-L and PickScore cosine scores.

Both component predictors return raw cosine similarities (comparable scale),
so a weighted sum is a well-conditioned joint objective; TFGFlowGuidance
rescales the total gradient norm as usual, keeping rho semantics unchanged.

Mirrors the reward ensemble the RL arm trains on
(reward_fn = {"clipscore": 0.5, "pickscore": 0.5}).
"""

from __future__ import annotations

import torch

from scripts.clip_predictor import CLIPPromptPredictor
from scripts.pickscore_predictor import PickScorePromptPredictor


class EnsemblePromptPredictor:
    def __init__(self, device: str | torch.device = "cuda",
                 w_clip: float = 0.5, w_pick: float = 0.5):
        self.clip = CLIPPromptPredictor(device=device)
        self.pick = PickScorePromptPredictor(device=device)
        self.w_clip = float(w_clip)
        self.w_pick = float(w_pick)
        self.device = self.clip.device
        self.dtype = self.clip.dtype

    def clear_text_cache(self):
        self.clip.clear_text_cache()
        self.pick.clear_text_cache()

    def __call__(self, image: torch.Tensor, prompt_text: list[str]) -> torch.Tensor:
        return (self.w_clip * self.clip(image, prompt_text)
                + self.w_pick * self.pick(image, prompt_text))
