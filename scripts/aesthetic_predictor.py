"""Differentiable LAION aesthetic-v2 predictor for TFG (MLP on CLIP-L embeds).
Interface matches CLIPPromptPredictor: (image[B,3,H,W] in [-1,1], prompts) -> score[B].
Prompt text is ignored (aesthetic is unconditional)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128),
            nn.Dropout(0.2), nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16),
            nn.Linear(16, 1))
    def forward(self, x): return self.layers(x)


class AestheticPredictor:
    def __init__(self, device="cuda", torch_dtype=torch.float32):
        from transformers import CLIPModel
        import urllib.request, pathlib, tempfile
        self.device = torch.device(device)
        self.dtype = torch_dtype
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                              torch_dtype=torch_dtype).to(self.device).eval()
        for p in self.clip.parameters(): p.requires_grad_(False)
        cache = pathlib.Path.home() / ".cache" / "aesthetic_v2.pth"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_URL, cache)
        self.mlp = _MLP().to(self.device, torch_dtype).eval()
        self.mlp.load_state_dict(torch.load(cache, map_location="cpu"))
        for p in self.mlp.parameters(): p.requires_grad_(False)
        self._mean = torch.tensor(_CLIP_MEAN, device=self.device, dtype=torch_dtype).view(1,3,1,1)
        self._std = torch.tensor(_CLIP_STD, device=self.device, dtype=torch_dtype).view(1,3,1,1)

    def clear_text_cache(self): pass

    def __call__(self, image, prompt_text=None):
        x = (image.clamp(-1,1) + 1) / 2
        x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
        x = (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)
        emb = self.clip.get_image_features(pixel_values=x.to(self.dtype))
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return self.mlp(emb).squeeze(-1) / 10.0  # /10 to keep near [0,1]
