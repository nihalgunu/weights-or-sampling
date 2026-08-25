#!/usr/bin/env python3
"""
GenEval-style detector-based scoring, without mmdetection.

Parses the study's GenEval-style prompt templates into structured checks and
scores each image with OWLv2 (zero-shot detector) + CLIP (crop color
classification):

  single object   -> object detected
  two objects     -> both detected
  counting        -> detected box count == N (IoU-NMS deduped)
  color(s)        -> object detected AND largest-crop CLIP color == target
  position        -> both detected AND box-center relation holds

Per image: strict 0/1 (all checks pass), GenEval convention. Reports per-arm
accuracy, paired delta vs no_lora, and per-category breakdown.

Usage (on the box):
    python3 scripts/cloud/detector_score.py \
        --image-root outputs/study_pickscore_pub_images \
        --prompts-file outputs/eval_prompts_pub.txt \
        --num-prompts 150 \
        --out outputs/study_pickscore_pub_detector.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

COLORS = ["red", "blue", "green", "yellow", "purple", "orange",
          "pink", "black", "white", "brown", "gray"]
NUMS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
POS = [(" above ", "above"), (" below ", "below"),
       (" to the left of ", "left"), (" to the right of ", "right"),
       (" on ", "above")]


def _strip_article(s: str) -> str:
    s = s.strip()
    for art in ("a ", "an "):
        if s.startswith(art):
            return s[len(art):]
    return s


def _parse_obj(s: str) -> dict:
    """'a red apple' -> {obj: 'apple', color: 'red'}; 'a bench' -> {obj: 'bench'}."""
    s = _strip_article(s)
    words = s.split()
    if len(words) >= 2 and words[0] in COLORS:
        return {"obj": " ".join(words[1:]), "color": words[0]}
    return {"obj": s, "color": None}


def _singular(s: str) -> str:
    if s.endswith("ies"):
        return s[:-3] + "y"
    if s.endswith("s") and not s.endswith("ss"):
        return s[:-1]
    return s


def parse_prompt(prompt: str) -> dict:
    p = prompt.strip().rstrip(".")
    prefix = "a photo of "
    if not p.startswith(prefix):
        return {"type": "unparsed", "prompt": prompt}
    rest = p[len(prefix):]
    for phrase, rel in POS:
        if phrase in rest:
            a, b = rest.split(phrase, 1)
            return {"type": "position", "objs": [_parse_obj(a), _parse_obj(b)], "rel": rel}
    # Depth relations (behind / in front of) are not verifiable from 2D boxes;
    # downgrade to a both-objects-present check (GenEval proper also only
    # scores left/right/above/below geometrically).
    for phrase in (" behind ", " in front of "):
        if phrase in rest:
            a, b = rest.split(phrase, 1)
            return {"type": "two", "objs": [_parse_obj(a), _parse_obj(b)]}
    if " and " in rest:
        a, b = rest.split(" and ", 1)
        return {"type": "two", "objs": [_parse_obj(a), _parse_obj(b)]}
    first = rest.split()[0]
    if first in NUMS:
        obj = _singular(" ".join(rest.split()[1:]))
        return {"type": "count", "objs": [{"obj": obj, "color": None}], "n": NUMS[first]}
    return {"type": "single", "objs": [_parse_obj(rest)]}


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def _nms(boxes, scores, thr=0.5):
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    for i in order:
        if all(_iou(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(i)
    return keep


class Scorer:
    def __init__(self, device="cuda", det_threshold=0.15):
        import torch
        from transformers import Owlv2Processor, Owlv2ForObjectDetection, CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.thr = det_threshold
        self.det_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.det = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble", torch_dtype=torch.float16).to(device).eval()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                              torch_dtype=torch.float16).to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self._color_text_cache = {}

    def detect(self, img, objs):
        """objs: list of names. Returns per-obj list of (box, score)."""
        queries = [[f"a photo of a {o}" for o in objs]]
        inputs = self.det_proc(text=queries, images=img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.det(**{k: (v.to(self.det.dtype) if v.dtype == self.torch.float32 else v)
                              for k, v in inputs.items()})
        res = self.det_proc.post_process_object_detection(
            out, threshold=self.thr,
            target_sizes=self.torch.tensor([img.size[::-1]]).to(self.device))[0]
        per_obj = {i: ([], []) for i in range(len(objs))}
        for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
            i = int(label)
            per_obj[i][0].append([float(x) for x in box])
            per_obj[i][1].append(float(score))
        result = []
        for i in range(len(objs)):
            boxes, scores = per_obj[i]
            keep = _nms(boxes, scores) if boxes else []
            result.append([(boxes[k], scores[k]) for k in keep])
        return result

    def crop_color(self, img, box, obj):
        x1, y1, x2, y2 = [max(0, v) for v in box]
        crop = img.crop((x1, y1, min(x2, img.width), min(y2, img.height)))
        if crop.width < 8 or crop.height < 8:
            return None
        key = obj
        if key not in self._color_text_cache:
            texts = [f"a {c} {obj}" for c in COLORS]
            tin = self.clip_proc(text=texts, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                tf = self.clip.get_text_features(**tin)
            self._color_text_cache[key] = tf / tf.norm(dim=-1, keepdim=True)
        tf = self._color_text_cache[key]
        iin = self.clip_proc(images=crop, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            vf = self.clip.get_image_features(pixel_values=iin["pixel_values"].to(self.clip.dtype))
        vf = vf / vf.norm(dim=-1, keepdim=True)
        return COLORS[int((vf @ tf.T).argmax())]

    def score(self, img, spec) -> dict:
        if spec["type"] == "unparsed":
            return {"correct": None}
        objs = [o["obj"] for o in spec["objs"]]
        dets = self.detect(img, objs)
        checks = {}
        if spec["type"] == "count":
            checks["count"] = len(dets[0]) == spec["n"]
        else:
            for i, o in enumerate(spec["objs"]):
                present = len(dets[i]) > 0
                checks[f"present_{o['obj']}"] = present
                if o["color"] and present:
                    best_box = max(dets[i], key=lambda t: t[1])[0]
                    checks[f"color_{o['obj']}"] = self.crop_color(img, best_box, o["obj"]) == o["color"]
                elif o["color"]:
                    checks[f"color_{o['obj']}"] = False
        if spec["type"] == "position":
            if all(len(d) > 0 for d in dets):
                (ax1, ay1, ax2, ay2) = max(dets[0], key=lambda t: t[1])[0]
                (bx1, by1, bx2, by2) = max(dets[1], key=lambda t: t[1])[0]
                acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
                bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                rel = spec["rel"]
                checks["position"] = {"above": acy < bcy, "below": acy > bcy,
                                      "left": acx < bcx, "right": acx > bcx}[rel]
            else:
                checks["position"] = False
        return {"correct": all(checks.values()), "checks": checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--num-prompts", type=int, default=150)
    ap.add_argument("--det-threshold", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from PIL import Image

    prompts = [ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()]
    prompts = prompts[:args.num_prompts]
    specs = [parse_prompt(p) for p in prompts]
    n_unparsed = sum(1 for s in specs if s["type"] == "unparsed")
    from collections import Counter
    print(f"[det] {len(prompts)} prompts; types: {Counter(s['type'] for s in specs)}; unparsed: {n_unparsed}")

    root = Path(args.image_root)
    arms = sorted([d.name for d in root.iterdir() if d.is_dir()])
    scorer = Scorer(det_threshold=args.det_threshold)
    results = {"meta": {"det_threshold": args.det_threshold, "num_prompts": len(prompts),
                        "types": {i: s["type"] for i, s in enumerate(specs)}, "arms": arms},
               "scores": {}}
    t0 = time.time()
    for arm in arms:
        arm_dir = root / arm
        seed_bases = sorted([d.name for d in arm_dir.iterdir() if d.is_dir()], key=int)
        arm_scores = {}
        for b in seed_bases:
            row = []
            for i, spec in enumerate(specs):
                p = arm_dir / b / f"{i:03d}.png"
                if not p.exists() or spec["type"] == "unparsed":
                    row.append(None)
                    continue
                try:
                    r = scorer.score(Image.open(p).convert("RGB"), spec)
                    row.append(1 if r["correct"] else 0)
                except Exception:
                    row.append(None)
            arm_scores[b] = row
        results["scores"][arm] = arm_scores
        flat = [v for b in arm_scores for v in arm_scores[b] if v is not None]
        print(f"[det] {arm}: acc={sum(flat)/len(flat):.3f} n={len(flat)} ({time.time()-t0:.0f}s)")
        Path(args.out).write_text(json.dumps(results, indent=1))

    # Paired summary vs no_lora (prompt-clustered)
    if "no_lora" in results["scores"]:
        base = results["scores"]["no_lora"]
        summary = {}
        for arm, sc in results["scores"].items():
            if arm == "no_lora":
                continue
            per_prompt = []
            for i in range(len(prompts)):
                ds = []
                for b in sc:
                    if b in base and sc[b][i] is not None and base[b][i] is not None:
                        ds.append(sc[b][i] - base[b][i])
                if ds:
                    per_prompt.append(sum(ds) / len(ds))
            if len(per_prompt) > 1:
                m = statistics.mean(per_prompt)
                se = statistics.stdev(per_prompt) / len(per_prompt) ** 0.5
                summary[arm] = {"d": m, "se": se, "t": m / se if se > 0 else 0.0,
                                "n": len(per_prompt)}
        results["summary"] = summary
        Path(args.out).write_text(json.dumps(results, indent=1))
        print("\n[det] paired accuracy delta vs no_lora (prompt-clustered):")
        for arm, e in summary.items():
            print(f"  {arm:28s} d={e['d']:+.4f} se={e['se']:.4f} t={e['t']:+.1f}")
    print(f"[det] wrote {args.out}")


if __name__ == "__main__":
    main()
