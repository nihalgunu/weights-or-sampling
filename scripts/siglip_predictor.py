"""
Differentiable SigLIP predictor for TFG-Flow (google/siglip-so400m-patch14-384).

A sigmoid-contrastive image-text model, trained differently from CLIP-L
(softmax contrastive) and PickScore/HPSv2 (preference fine-tunes), so it is a
distinct point on the reward axis. The study already uses SigLIP as an
external cross-check scorer (external_metric.py); this class makes it a
guidable / trainable reward with the standard interface:

    predictor(image[B,3,H,W] in [-1,1], prompt_text: list[str]) -> score[B]

Returns the raw cosine similarity (SigLIP's logit = scale*cos + bias; the
affine map does not change the gradient direction and TFG rescales norms).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class SiglipPromptPredictor:
    def __init__(
        self,
        device: str | torch.device = "cuda",
        model_name: str = "google/siglip-so400m-patch14-384",
        torch_dtype: torch.dtype = torch.float32,
    ):
        from transformers import AutoProcessor, SiglipModel

        self.device = torch.device(device)
        self.dtype = torch_dtype
        self.model = SiglipModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        proc = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = proc.tokenizer
        ip = proc.image_processor
        size = ip.size
        if isinstance(size, dict):
            size = (size.get("height") or size.get("shortest_edge"), size.get("width") or size.get("shortest_edge"))
        self.input_size = tuple(int(s) for s in size)
        self._mean = torch.tensor(ip.image_mean, device=self.device, dtype=torch_dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(ip.image_std, device=self.device, dtype=torch_dtype).view(1, 3, 1, 1)
        self._text_cache: dict[tuple, torch.Tensor] = {}

    def _encode_text(self, prompt_text: list[str]) -> torch.Tensor:
        key = tuple(prompt_text)
        if key in self._text_cache:
            return self._text_cache[key]
        # SigLIP was trained with max_length padding (64 tokens).
        inputs = self.tokenizer(list(prompt_text), padding="max_length", max_length=64,
                                truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            t = self.model.get_text_features(**inputs)
        t = F.normalize(t.to(self.dtype), dim=-1)
        self._text_cache[key] = t
        return t

    def clear_text_cache(self) -> None:
        self._text_cache.clear()

    def __call__(self, image: torch.Tensor, prompt_text: list[str]) -> torch.Tensor:
        x = (image.clamp(-1.0, 1.0) + 1.0) / 2.0
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        x = (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)
        i = F.normalize(self.model.get_image_features(pixel_values=x.to(self.dtype)), dim=-1)
        t = self._encode_text(prompt_text)
        if i.shape[0] != t.shape[0]:
            if t.shape[0] == 1:
                t = t.expand(i.shape[0], -1)
            else:
                raise ValueError(f"image batch {i.shape[0]} vs text batch {t.shape[0]}")
        return (i * t).sum(dim=-1)
