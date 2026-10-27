#!/usr/bin/env python3
"""
Exact-state checkpointing for the LoRA trainers (train_draft.py,
train_distillation.py), so a preempted or lost box can resume a run that is
indistinguishable from an uninterrupted one.

Why weight-only checkpoints are not enough
------------------------------------------
The DRaFT result is a *stability* story (collapse at a stochastic step). A
resume that resets Adam moments, the prompt sampler, or the noise stream changes
the dynamics, so a preempted-and-resumed arm would be a different experiment.
This module therefore saves, next to the usual <ckpt>/lora directory:

    <ckpt>/train_state.pt
        trainable   {param_name: tensor}      every requires_grad parameter
        optimizer   optimizer.state_dict()    Adam moments + step counts
        rng         python Random state, torch CPU state, torch CUDA states
        loop        step, micro, elapsed_s, reward/loss history, guard state
        config      the training-relevant cfg keys (checked on resume)

The file is written to a temp name and renamed, so a preemption mid-save leaves
either the previous complete state or the new one, never a torn file. The LoRA
directory is still saved first (for eval), but resume restores the trainable
parameters from train_state.pt, so a partially synced lora/ dir cannot corrupt
a resumed run.

Equivalence is verified on CPU stubs by tests/test_train_draft_smoke.py
(test_resume_matches_uninterrupted) and tests/test_distillation_smoke.py.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Iterable

import torch

TRAIN_STATE_FILE = "train_state.pt"


def trainable_named(module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {n: p for n, p in module.named_parameters() if p.requires_grad}


def capture_rng(py_rng: random.Random | None = None) -> dict:
    state = {
        "python_global": random.getstate(),
        "python_local": py_rng.getstate() if py_rng is not None else None,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def restore_rng(state: dict, py_rng: random.Random | None = None) -> None:
    random.setstate(state["python_global"])
    if py_rng is not None and state.get("python_local") is not None:
        py_rng.setstate(state["python_local"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except RuntimeError as e:  # different device count after re-provisioning
            print(f"[train_state] WARN: could not restore CUDA RNG ({e}); noise is generator-seeded anyway")


def save_train_state(
    ckpt_dir: str | Path,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loop: dict,
    config: dict,
    py_rng: random.Random | None = None,
    keep_states: int = 2,
) -> Path:
    """Atomic write of <ckpt_dir>/train_state.pt. `loop` is the trainer's own
    counters/history (JSON-serialisable); `config` the keys checked on resume."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trainable": {n: p.detach().to("cpu").clone() for n, p in trainable_named(module).items()},
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng(py_rng),
        "loop": loop,
        "config": config,
    }
    final = ckpt_dir / TRAIN_STATE_FILE
    tmp = ckpt_dir / (TRAIN_STATE_FILE + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, final)
    prune_old_states(ckpt_dir.parent, keep=keep_states)
    return final


def prune_old_states(output_dir: Path, keep: int = 2) -> list[Path]:
    """Keep train_state.pt (LoRA + Adam moments, ~3x the adapter) only on the
    newest `keep` checkpoints; the lora/ dirs stay for eval. Returns removed."""
    states = sorted((d for d in Path(output_dir).iterdir()
                     if d.is_dir() and _ckpt_num(d) >= 0 and (d / TRAIN_STATE_FILE).is_file()),
                    key=_ckpt_num)
    removed = []
    for d in states[:-keep] if keep > 0 else states:
        (d / TRAIN_STATE_FILE).unlink()
        removed.append(d / TRAIN_STATE_FILE)
    return removed


def load_train_state(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _ckpt_num(p: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", p.name)
    return int(m.group(1)) if m else -1


def find_resume_checkpoint(output_dir: str | Path) -> Path | None:
    """Highest-numbered <output_dir>/checkpoint-N that has a complete train_state.pt."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    cands = [d for d in output_dir.iterdir()
             if d.is_dir() and _ckpt_num(d) >= 0 and (d / TRAIN_STATE_FILE).is_file()]
    if not cands:
        return None
    return max(cands, key=_ckpt_num)


def resolve_resume(resume: str | Path | None, output_dir: str | Path) -> Path | None:
    """'auto' -> latest resumable checkpoint under output_dir (None if fresh);
    'none'/None/'' -> fresh; otherwise an explicit checkpoint dir (must exist)."""
    if resume is None or str(resume).lower() in ("", "none", "off", "false"):
        return None
    if str(resume).lower() == "auto":
        return find_resume_checkpoint(output_dir)
    p = Path(resume)
    if not (p / TRAIN_STATE_FILE).is_file():
        raise FileNotFoundError(f"--resume {p}: no {TRAIN_STATE_FILE} there")
    return p


def check_config(saved: dict, current: dict, keys: Iterable[str]) -> None:
    """Refuse to resume a run whose training-relevant config changed; a silent
    resume under a new lr/batch would be a different experiment with the old name."""
    bad = {}
    for k in keys:
        a, b = saved.get(k), current.get(k)
        if a != b:
            bad[k] = (a, b)
    if bad:
        raise RuntimeError(f"resume refused: config differs from checkpoint: {bad}")


def restore_trainable(module: torch.nn.Module, saved: dict[str, torch.Tensor]) -> int:
    live = trainable_named(module)
    if set(live) != set(saved):
        missing = sorted(set(saved) - set(live))[:5]
        extra = sorted(set(live) - set(saved))[:5]
        raise RuntimeError(f"resume refused: trainable parameter names differ "
                           f"(missing {missing}, extra {extra})")
    with torch.no_grad():
        for n, p in live.items():
            p.copy_(saved[n].to(device=p.device, dtype=p.dtype))
    return len(live)


def truncate_jsonl(path: str | Path, max_step: int, key: str = "step") -> int:
    """Drop log records written after the checkpoint we are resuming from (a run
    preempted between a checkpoint and the next one has logged steps that will
    now be replayed). Returns the number of records kept."""
    path = Path(path)
    if not path.is_file():
        return 0
    kept = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if int(rec.get(key, 0)) <= max_step:
            kept.append(ln)
    path.write_text("".join(k + "\n" for k in kept))
    return len(kept)


def save_lora_dir(transformer: torch.nn.Module, lora_dir: str | Path) -> Path:
    """<ckpt>/lora in the FlowGRPO layout; falls back to a state_dict for stubs."""
    d = Path(lora_dir)
    d.mkdir(parents=True, exist_ok=True)
    if hasattr(transformer, "save_pretrained"):
        transformer.save_pretrained(str(d))
    else:
        torch.save(transformer.state_dict(), d / "state_dict.pt")
    return d


def config_subset(cfg, keys: Iterable[str]) -> dict:
    d = vars(cfg) if not isinstance(cfg, dict) else cfg
    return {k: d.get(k) for k in keys}
