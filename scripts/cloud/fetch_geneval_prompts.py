#!/usr/bin/env python3
"""
Fetch GenEval's evaluation_metadata.jsonl and build a distillation-train pool
disjoint from our 30-prompt eval set.

Train pool is used only for TFG-teacher dataset generation. Eval pool is the
existing scripts/cloud/prompts_smoke.json and stays untouched — that's what we
measure Cells A, C', D' on, giving a clean held-out test of whether the
distilled model generalizes rather than memorizes the training prompts.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
from pathlib import Path
from urllib.request import urlopen

# Upstream GenEval prompts — used for the official benchmark, ~553 prompts with
# structured metadata. We only need the prompt strings for distillation training.
GENEVAL_URL = (
    "https://raw.githubusercontent.com/djghosh13/geneval/main/"
    "prompts/evaluation_metadata.jsonl"
)


def _fetch(source: str) -> str:
    """Load JSONL from either a URL (http/https) or a local path.

    System Python on macOS ships without root CAs, which makes urllib fail on
    HTTPS. Fall back to an unverified SSL context if the default one chokes —
    these are public GitHub files, the threat model is already 'trust what's
    in the repo.'
    """
    if source.startswith(("http://", "https://")):
        try:
            with urlopen(source, timeout=30) as resp:
                return resp.read().decode()
        except Exception as e:
            print(f"default urlopen failed ({e!r}); retrying with unverified ctx")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urlopen(source, timeout=30, context=ctx) as resp:
                return resp.read().decode()
    return Path(source).read_text()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="scripts/cloud/prompts_train.json")
    parser.add_argument("--eval_pool", default="scripts/cloud/prompts_smoke.json",
                        help="Prompts to exclude from the train pool (held-out set).")
    parser.add_argument("--max_prompts", type=int, default=200,
                        help="Cap the train pool size. 200 is enough distillation "
                             "signal for a 500-step LoRA finetune; more gives "
                             "diminishing returns and more compute for Stage 1.")
    parser.add_argument("--source", default=GENEVAL_URL,
                        help="URL (https) or local path to evaluation_metadata.jsonl.")
    args = parser.parse_args()

    # Load held-out eval prompts — lowercase + stripped for robust matching.
    eval_texts = set()
    if Path(args.eval_pool).exists():
        with open(args.eval_pool) as f:
            for p in json.load(f):
                eval_texts.add(p["prompt"].strip().lower())
    print(f"loaded {len(eval_texts)} held-out eval prompts")

    # Fetch GenEval metadata. Each line is a JSON object with at least "prompt".
    print(f"fetching {args.source}")
    raw = _fetch(args.source)

    train = []
    seen = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = obj.get("prompt")
        if not prompt:
            continue
        key = prompt.strip().lower()
        if key in eval_texts or key in seen:
            continue
        seen.add(key)
        train.append({
            "id": len(train),
            "prompt": prompt,
            "tag": obj.get("tag", ""),
        })
        if len(train) >= args.max_prompts:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(train, f, indent=2)

    print(f"wrote {len(train)} training prompts to {out_path}")
    # Sanity check: no leakage into eval set.
    leaked = [p for p in train if p["prompt"].strip().lower() in eval_texts]
    assert not leaked, f"{len(leaked)} train prompts overlap the eval pool"
    print("no overlap with eval pool — held-out test is clean")


if __name__ == "__main__":
    main()
