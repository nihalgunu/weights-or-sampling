"""Differentiable ImageReward predictor for TFG (BLIP-based; distinct model
family from CLIP/SigLIP/PickScore). Uses ImageReward's ReFL-style score_gard
path: BLIP image encoder + cross-attention text encoder + MLP head, all
differentiable w.r.t. the input image tensor."""
from __future__ import annotations
import torch
import torch.nn.functional as F

_IR_MEAN = (0.48145466, 0.4578275, 0.40821073)
_IR_STD = (0.26862954, 0.26130258, 0.27577711)


class ImageRewardPredictor:
    def __init__(self, device="cuda"):
        import ImageReward as RM
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.rm = RM.load("ImageReward-v1.0", device=str(device))
        self.rm.eval()
        for p in self.rm.parameters(): p.requires_grad_(False)
        self._mean = torch.tensor(_IR_MEAN, device=self.device).view(1,3,1,1)
        self._std = torch.tensor(_IR_STD, device=self.device).view(1,3,1,1)
        self._tok_cache = {}

    def clear_text_cache(self): self._tok_cache.clear()

    def _tok(self, prompts):
        key = tuple(prompts)
        if key not in self._tok_cache:
            self._tok_cache[key] = self.rm.blip.tokenizer(
                list(prompts), padding="max_length", truncation=True,
                max_length=35, return_tensors="pt").to(self.device)
        return self._tok_cache[key]

    def __call__(self, image, prompt_text):
        x = (image.clamp(-1,1) + 1) / 2
        x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
        x = (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)
        if x.shape[0] != len(prompt_text) and len(prompt_text) == 1:
            prompt_text = list(prompt_text) * x.shape[0]
        t = self._tok(prompt_text)
        # ReFL score path (differentiable)
        image_embeds = self.rm.blip.visual_encoder(x.float())
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
        text_output = self.rm.blip.text_encoder(
            t.input_ids, attention_mask=t.attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,
            return_dict=True)
        txt_feat = text_output.last_hidden_state[:, 0, :].float()
        rewards = self.rm.mlp(txt_feat).squeeze(-1)
        return (rewards - self.rm.mean) / self.rm.std / 10.0
