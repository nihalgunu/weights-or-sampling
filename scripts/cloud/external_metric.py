#!/usr/bin/env python3
"""
External (non-optimized) metrics over the saved study images.

Breaks the evaluation-reflexivity caveat: neither RL nor TFG ever optimized
these models.

  siglip : SigLIP text-image score (google/siglip-so400m-patch14-384) — a
           different model family / training objective from both CLIP-L and
           PickScore (CLIP-H).
  ocr    : EasyOCR text match — for OCR-set prompts containing a quoted target
           string, normalized similarity between the target and the detected
           text. Fully independent of any embedding model.

Reads the eval matrix's image tree: <image-root>/<arm>/<seed_base>/<idx>.png
Writes per-arm per-seed per-prompt scores + paired summaries vs no_lora.

Usage (on the box, after eval_study_matrix.py):
    python3 scripts/cloud/external_metric.py \
        --image-root outputs/study_clip_images \
        --prompts-file outputs/eval_prompts_pub.txt \
        --metrics siglip \
        --out outputs/study_clip_external.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path


def load_prompts(path: Path, n: int) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return lines[:n]


class SiglipScorer:
    def __init__(self, device="cuda", model_name="google/siglip-so400m-patch14-384"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self._text_cache = {}

    def text_feats(self, prompt):
        if prompt not in self._text_cache:
            inp = self.processor(text=[prompt], padding="max_length", truncation=True,
                                 return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                t = self.model.get_text_features(**inp)
            self._text_cache[prompt] = t / t.norm(dim=-1, keepdim=True)
        return self._text_cache[prompt]

    def __call__(self, img, prompt) -> float:
        inp = self.processor(images=[img], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            v = self.model.get_image_features(pixel_values=inp["pixel_values"].to(self.model.dtype))
        v = v / v.norm(dim=-1, keepdim=True)
        return float((v @ self.text_feats(prompt).T).item())


class OcrScorer:
    """Similarity between the quoted target text in the prompt and what EasyOCR
    reads off the image. 0 if the prompt has no quoted string (scored as None)."""

    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    @staticmethod
    def target(prompt: str):
        m = re.findall(r'["“]([^"”]+)["”]', prompt)
        return m[0].strip().lower() if m else None

    def __call__(self, img_path: str, prompt: str):
        from difflib import SequenceMatcher
        tgt = self.target(prompt)
        if not tgt:
            return None
        try:
            det = self.reader.readtext(img_path, detail=0)
        except Exception:
            return None
        if not det:
            return 0.0
        joined = " ".join(d.lower() for d in det)
        best = SequenceMatcher(None, tgt, joined).ratio()
        for d in det:
            best = max(best, SequenceMatcher(None, tgt, d.lower()).ratio())
        return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--num-prompts", type=int, default=150)
    ap.add_argument("--metrics", default="siglip", help="comma list: siglip,ocr")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from PIL import Image

    prompts = load_prompts(Path(args.prompts_file), args.num_prompts)
    metrics = [m for m in args.metrics.split(",") if m]
    root = Path(args.image_root)
    arms = sorted([d.name for d in root.iterdir() if d.is_dir()])
    print(f"[ext] arms: {arms}; metrics: {metrics}; prompts: {len(prompts)}")

    scorers = {}
    if "siglip" in metrics:
        scorers["siglip"] = SiglipScorer()
    if "ocr" in metrics:
        scorers["ocr"] = OcrScorer()

    results = {"meta": {"metrics": metrics, "num_prompts": len(prompts), "arms": arms},
               "scores": {}}
    t0 = time.time()
    for arm in arms:
        arm_dir = root / arm
        seed_bases = sorted([d.name for d in arm_dir.iterdir() if d.is_dir()], key=int)
        arm_scores = {m: {} for m in metrics}
        try:
            for b in seed_bases:
                for m in metrics:
                    arm_scores[m][b] = []
                for i, prompt in enumerate(prompts):
                    p = arm_dir / b / f"{i:03d}.png"
                    if not p.exists():
                        for m in metrics:
                            arm_scores[m][b].append(None)
                        continue
                    img = Image.open(p).convert("RGB")
                    if "siglip" in scorers:
                        arm_scores["siglip"][b].append(scorers["siglip"](img, prompt))
                    if "ocr" in scorers:
                        arm_scores["ocr"][b].append(scorers["ocr"](str(p), prompt))
        except Exception as e:
            print(f"[ext] WARN: arm {arm} failed: {e}")
        results["scores"][arm] = arm_scores
        done = {m: sum(1 for b in arm_scores[m] for v in arm_scores[m][b] if v is not None)
                for m in metrics}
        print(f"[ext] {arm}: scored {done} ({time.time()-t0:.0f}s)")
        Path(args.out).write_text(json.dumps(results, indent=1))

    # Paired summary vs no_lora
    if "no_lora" in results["scores"]:
        base = results["scores"]["no_lora"]
        summary = {}
        for arm, sc in results["scores"].items():
            if arm == "no_lora":
                continue
            summary[arm] = {}
            for m in metrics:
                deltas = []
                for b in sc.get(m, {}):
                    if b not in base.get(m, {}):
                        continue
                    for a, c in zip(sc[m][b], base[m][b]):
                        if a is not None and c is not None:
                            deltas.append(a - c)
                if len(deltas) > 1:
                    dm = statistics.mean(deltas)
                    se = statistics.stdev(deltas) / len(deltas) ** 0.5
                    summary[arm][m] = {"d": dm, "se": se,
                                       "z": dm / se if se > 0 else 0.0,
                                       "wins": sum(1 for x in deltas if x > 0),
                                       "n": len(deltas)}
        results["summary"] = summary
        Path(args.out).write_text(json.dumps(results, indent=1))
        print("\n[ext] paired summary vs no_lora:")
        for arm, ms in summary.items():
            for m, e in ms.items():
                print(f"  {arm:34s} {m:7s} d={e['d']:+.4f} z={e['z']:+.1f} wins={e['wins']}/{e['n']}")

    print(f"[ext] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
