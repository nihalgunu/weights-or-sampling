"""
Differentiable HPSv2 predictor for TFG-Flow.

HPSv2 (Human Preference Score v2, Wu et al. 2023) is an open_clip ViT-H/14
fine-tuned on human preference data. Like PickScore it is a contrastive
image-text model, so it serves as a TFG guidance predictor the same way
CLIPPromptPredictor / PickScorePromptPredictor do, and as a fresh point on the
prompt-dependence axis (a differentiable reward from a distinct training corpus).

Interface matches the CLIP/PickScore predictors exactly:
    predictor(image[B,3,H,W] in [-1,1], prompt_text: list[str]) -> score[B]
Returns raw cosine similarity (TFGFlowGuidance rescales gradient norms, so rho
keeps its usual semantics).

Loading path: open_clip create_model('ViT-H-14') + the HPS_v2.1 checkpoint
(HF repo xswu/HPSv2). We forward through encode_image/encode_text with grads on
the image only (all params frozen).
"""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F


class HPSv2PromptPredictor:
    def __init__(
        self,
        device: str | torch.device = "cuda",
        model_arch: str = "ViT-H-14",
        ckpt_name: str = "HPS_v2.1_compressed.pt",
        hf_repo: str = "xswu/HPSv2",
        torch_dtype: torch.dtype = torch.float32,
    ):
        import open_clip
        from huggingface_hub import hf_hub_download

        self.device = torch.device(device)
        self.dtype = torch_dtype

        model, _, _ = open_clip.create_model_and_transforms(
            model_arch, pretrained=None, precision="fp32", device=str(self.device)
        )
        ckpt_path = hf_hub_download(repo_id=hf_repo, filename=ckpt_name)
        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("state_dict", state)
        # strip a possible 'module.' prefix
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        # tokenizer for the same arch
        self.tokenizer = open_clip.get_tokenizer(model_arch)

        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # open_clip ViT-H/14 preprocessing: 224px, OpenAI CLIP normalization
        self.input_size = (224, 224)
        mean = torch.tensor((0.48145466, 0.4578275, 0.40821073),
                            device=self.device, dtype=torch_dtype).view(1, 3, 1, 1)
        std = torch.tensor((0.26862954, 0.26130258, 0.27577711),
                           device=self.device, dtype=torch_dtype).view(1, 3, 1, 1)
        self._mean = mean
        self._std = std
        self._text_cache: dict[tuple, torch.Tensor] = {}

    def _encode_text(self, prompt_text: list[str]) -> torch.Tensor:
        key = tuple(prompt_text)
        if key in self._text_cache:
            return self._text_cache[key]
        tokens = self.tokenizer(list(prompt_text)).to(self.device)
        with torch.no_grad():
            text_feats = self.model.encode_text(tokens)
        text_feats = F.normalize(text_feats.to(self.dtype), dim=-1)
        self._text_cache[key] = text_feats
        return text_feats

    def clear_text_cache(self) -> None:
        self._text_cache.clear()

    def __call__(self, image: torch.Tensor, prompt_text: list[str]) -> torch.Tensor:
        """image: (B,3,H,W) in [-1,1] (VAE decode output). Returns (B,) cosine sim."""
        if image.shape[0] != len(prompt_text) and not (
            image.shape[0] == 1 and len(prompt_text) == 1
        ):
            raise ValueError(f"image batch {image.shape[0]} != prompt count {len(prompt_text)}")

        image_01 = (image.clamp(-1.0, 1.0) + 1.0) / 2.0
        image_in = F.interpolate(image_01, size=self.input_size, mode="bilinear", align_corners=False)
        image_in = (image_in - self._mean.to(image_in.dtype)) / self._std.to(image_in.dtype)

        text_feats = self._encode_text(prompt_text)
        image_feats = self.model.encode_image(image_in.to(self.dtype))
        image_feats = F.normalize(image_feats, dim=-1)

        if image_feats.shape[0] != text_feats.shape[0]:
            if text_feats.shape[0] == 1:
                text_feats = text_feats.expand(image_feats.shape[0], -1)
            else:
                raise ValueError(f"image batch {image_feats.shape[0]} vs text batch {text_feats.shape[0]}")
        return (image_feats * text_feats).sum(dim=-1)
