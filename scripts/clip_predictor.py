"""
Differentiable CLIP-based prompt-image similarity predictor for TFG-Flow.

Why CLIP and not GenEval directly:
    GenEval scores via mmdetection-based object counting / attribute checks.
    That pipeline is not differentiable — you cannot backprop through "is there
    a red car in this image?" The upstream TFG paper faces the same issue and
    uses differentiable classifiers (CLIP-class-logits) in its image_label_guidance
    task. We do the analogous thing: CLIP-text-image cosine similarity as the
    predictor, and keep GenEval strictly as the downstream *evaluation* metric.

    The guidance signal is therefore "shift x_0 toward higher CLIP-prompt match".
    This is a proxy for GenEval, correlated but imperfect. Any reported gain must
    be attributed carefully: TFG+CLIP-predictor improves GenEval_X by Y. That is
    the honest claim.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPPromptPredictor:
    def __init__(
        self,
        device: str | torch.device = "cuda",
        model_name: str = "openai/clip-vit-large-patch14",
        torch_dtype: torch.dtype = torch.float32,
    ):
        from transformers import CLIPModel, CLIPTokenizer

        self.device = torch.device(device)
        self.dtype = torch_dtype
        self.model = (
            CLIPModel.from_pretrained(model_name, torch_dtype=torch_dtype)
            .to(self.device)
            .eval()
        )
        for p in self.model.parameters():
            p.requires_grad_(False)  # predictor is frozen; we only need grads thru inputs
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        mean = torch.tensor(_CLIP_MEAN, device=self.device, dtype=torch_dtype).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD, device=self.device, dtype=torch_dtype).view(1, 3, 1, 1)
        self._mean = mean
        self._std = std

        self._text_cache: dict[tuple, torch.Tensor] = {}

    def _encode_text(self, prompt_text: list[str]) -> torch.Tensor:
        """Text embeddings (B, D), cached by prompt tuple.

        Text features don't depend on the image, so we cache them across all
        scheduler steps within a single pipe(...) call. Callers that reuse this
        predictor across many prompts will see the cache grow; it's cleared per
        pipe call if you construct a new predictor, which we don't.
        """
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

    def __call__(
        self, image: torch.Tensor, prompt_text: list[str]
    ) -> torch.Tensor:
        """Score a batch of images against their prompts.

        image: (B, 3, H, W) in approximately [-1, 1] (VAE decode output).
        prompt_text: list[str] of length B.
        Returns: (B,) similarity in roughly [-1, 1]; higher is better alignment.
        """
        if image.shape[0] != len(prompt_text):
            # Pipeline runs one prompt at a time by default; broadcast if image
            # batch is 1 and prompts is 1 (trivial case) — otherwise shape error.
            if image.shape[0] == 1 and len(prompt_text) == 1:
                pass
            else:
                raise ValueError(
                    f"image batch {image.shape[0]} != prompt count {len(prompt_text)}"
                )

        # VAE decode gives [-1, 1]; CLIP wants [0, 1] pre-normalization.
        image_01 = (image.clamp(-1.0, 1.0) + 1.0) / 2.0
        # Differentiable resize; bilinear preserves grad path.
        image_224 = F.interpolate(
            image_01, size=(224, 224), mode="bilinear", align_corners=False
        )
        image_clip = (image_224 - self._mean.to(image_224.dtype)) / self._std.to(image_224.dtype)

        text_feats = self._encode_text(prompt_text)  # (B_txt, D)
        image_feats = self.model.get_image_features(pixel_values=image_clip.to(self.dtype))
        image_feats = F.normalize(image_feats, dim=-1)

        # Align dims: if B_img > B_txt, tile; else element-wise.
        if image_feats.shape[0] != text_feats.shape[0]:
            if text_feats.shape[0] == 1:
                text_feats = text_feats.expand(image_feats.shape[0], -1)
            else:
                raise ValueError(
                    f"image batch {image_feats.shape[0]} vs text batch {text_feats.shape[0]}"
                )
        sim = (image_feats * text_feats).sum(dim=-1)  # (B,)
        return sim
