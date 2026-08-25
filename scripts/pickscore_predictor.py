"""
Differentiable PickScore predictor for TFG-Flow.

PickScore_v1 (yuvalkirstain/PickScore_v1) is a CLIP-ViT-H fine-tuned on human
preference pairs (Pick-a-Pic). It is fully differentiable, so it can serve as a
TFG guidance predictor the same way CLIPPromptPredictor does — which is what a
fair "TFG vs RL, same individual reward" comparison requires: FlowGRPO trains
on PickScore, so TFG must guide on PickScore.

Interface matches CLIPPromptPredictor exactly:
    predictor(image[B,3,H,W] in [-1,1], prompt_text: list[str]) -> score[B]
Returns the raw cosine similarity (not logit-scaled). TFGFlowGuidance rescales
gradient norms (`rescale_grad`), so the guidance strength ρ has the same
semantics as with the CLIP predictor. Offline eval scoring stays on the
flow_grpo scale (logit_scale * cos / 26) via PickScoreScorer — scale only
affects reporting, not the gradient direction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PickScorePromptPredictor:
    def __init__(
        self,
        device: str | torch.device = "cuda",
        model_name: str = "yuvalkirstain/PickScore_v1",
        processor_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype: torch.dtype = torch.float32,
    ):
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.dtype = torch_dtype
        self.model = (
            CLIPModel.from_pretrained(model_name, torch_dtype=torch_dtype)
            .to(self.device)
            .eval()
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

        processor = CLIPProcessor.from_pretrained(processor_name)
        self.tokenizer = processor.tokenizer
        ip = processor.image_processor
        # Read normalization + input size from the processor config rather than
        # hardcoding — ViT-H's constants happen to match OpenAI CLIP's, but this
        # keeps the predictor correct if the processor ever differs.
        size = ip.crop_size if hasattr(ip, "crop_size") else ip.size
        if isinstance(size, dict):
            size = (size.get("height") or size.get("shortest_edge"),
                    size.get("width") or size.get("shortest_edge"))
        self.input_size = tuple(int(s) for s in size)
        mean = torch.tensor(ip.image_mean, device=self.device, dtype=torch_dtype).view(1, 3, 1, 1)
        std = torch.tensor(ip.image_std, device=self.device, dtype=torch_dtype).view(1, 3, 1, 1)
        self._mean = mean
        self._std = std

        self._text_cache: dict[tuple, torch.Tensor] = {}

    def _encode_text(self, prompt_text: list[str]) -> torch.Tensor:
        key = tuple(prompt_text)
        if key in self._text_cache:
            return self._text_cache[key]
        inputs = self.tokenizer(
            list(prompt_text),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            text_feats = self.model.get_text_features(**inputs)
        text_feats = F.normalize(text_feats.to(self.dtype), dim=-1)
        self._text_cache[key] = text_feats
        return text_feats

    def clear_text_cache(self) -> None:
        self._text_cache.clear()

    def __call__(self, image: torch.Tensor, prompt_text: list[str]) -> torch.Tensor:
        """image: (B, 3, H, W) in [-1, 1] (VAE decode output). Returns (B,) cosine sim."""
        if image.shape[0] != len(prompt_text) and not (
            image.shape[0] == 1 and len(prompt_text) == 1
        ):
            raise ValueError(
                f"image batch {image.shape[0]} != prompt count {len(prompt_text)}"
            )

        image_01 = (image.clamp(-1.0, 1.0) + 1.0) / 2.0
        image_in = F.interpolate(
            image_01, size=self.input_size, mode="bilinear", align_corners=False
        )
        image_in = (image_in - self._mean.to(image_in.dtype)) / self._std.to(image_in.dtype)

        text_feats = self._encode_text(prompt_text)
        image_feats = self.model.get_image_features(pixel_values=image_in.to(self.dtype))
        image_feats = F.normalize(image_feats, dim=-1)

        if image_feats.shape[0] != text_feats.shape[0]:
            if text_feats.shape[0] == 1:
                text_feats = text_feats.expand(image_feats.shape[0], -1)
            else:
                raise ValueError(
                    f"image batch {image_feats.shape[0]} vs text batch {text_feats.shape[0]}"
                )
        return (image_feats * text_feats).sum(dim=-1)
