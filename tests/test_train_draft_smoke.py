"""
CPU smoke test for scripts/train_draft.py (the DRaFT arm).

Reuses the SD3/VAE/predictor stubs from test_train_ddrl_smoke.py. Checks the
plumbing that would otherwise only fail on a paid GPU: gradient reaches the
policy, the loop learns on a fixed objective, budget stops (steps / wall
clock) fire, grad accumulation counts, checkpoints land in the FlowGRPO
layout (<run>/checkpoint-N/lora), and the JSON-lines log is written.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_train_ddrl_smoke import (  # noqa: E402
    StubPredictor,
    _fake_embeds,
    _make_stub_pipe,
)
from scripts.train_draft import draft_step, load_prompts, train_draft  # noqa: E402


def _cfg(**over):
    base = dict(
        seed=3, lr=1e-2, weight_decay=0.0, batch_size=2, grad_accum=1, grad_steps=1,
        num_inference_steps=4, guidance_scale=4.5, height=8, width=8, kl_coef=0.0,
        max_steps=None, wall_hours=None, max_grad_norm=1.0, save_every=0,
        save_every_minutes=0.0, log_every=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _encode(texts):
    return _fake_embeds(len(texts))


def test_draft_step_grad_reaches_policy():
    torch.manual_seed(1)
    pipe = _make_stub_pipe(batch=2)
    B, C, H, W = pipe._shape
    cfg = _cfg()
    gen = torch.Generator().manual_seed(0)
    loss, r, kl = draft_step(pipe, StubPredictor(), ["a", "b"], _encode(["a", "b"]),
                             (B, C, H, W), cfg, gen, "cpu")
    assert torch.isfinite(loss) and r.dim() == 0 and kl.item() == 0.0
    loss.backward()
    g = pipe.transformer.conv.weight.grad
    assert g is not None and g.abs().sum().item() > 0, "no gradient reached the policy"


def test_loop_learns_and_writes_artifacts():
    """Predictor = mean pixel; 25 AdamW steps on a 39-param conv must raise it."""
    torch.manual_seed(2)
    pipe = _make_stub_pipe(batch=2)
    with tempfile.TemporaryDirectory() as d:
        s = train_draft(pipe, StubPredictor(), ["p1", "p2", "p3"], _cfg(max_steps=25),
                        d, device="cpu", encode_fn=_encode, log_fn=lambda *_: None)
        assert s["steps"] == 25 and s["stop_reason"] == "max_steps"
        assert s["reward_last10"] > s["reward_first10"], (
            f"reward did not rise: {s['reward_first10']:.4f} -> {s['reward_last10']:.4f}")
        lines = Path(d, "train_log.jsonl").read_text().splitlines()
        assert len(lines) == 25
        rec = json.loads(lines[-1])
        assert rec["step"] == 25 and "reward" in rec and "elapsed_s" in rec
        assert Path(d, "checkpoint-25", "lora").is_dir(), "final checkpoint not in <run>/checkpoint-N/lora"
        assert json.loads(Path(d, "summary.json").read_text())["steps"] == 25
    print(f"  [draft-learn] {s['reward_first10']:.4f} -> {s['reward_last10']:.4f}")


def test_wall_clock_budget_stops():
    pipe = _make_stub_pipe(batch=2)
    with tempfile.TemporaryDirectory() as d:
        s = train_draft(pipe, StubPredictor(), ["p"], _cfg(wall_hours=1e-7),
                        d, device="cpu", encode_fn=_encode, log_fn=lambda *_: None)
    assert s["stop_reason"] == "wall_hours" and s["steps"] <= 2, s


def test_grad_accum_counts_micro_batches():
    pipe = _make_stub_pipe(batch=2)
    with tempfile.TemporaryDirectory() as d:
        s = train_draft(pipe, StubPredictor(), ["p"], _cfg(max_steps=3, grad_accum=2),
                        d, device="cpu", encode_fn=_encode, log_fn=lambda *_: None)
    assert s["steps"] == 3 and s["micro_batches"] == 6, s


def test_periodic_checkpoints_by_step():
    pipe = _make_stub_pipe(batch=2)
    with tempfile.TemporaryDirectory() as d:
        train_draft(pipe, StubPredictor(), ["p"], _cfg(max_steps=4, save_every=2),
                    d, device="cpu", encode_fn=_encode, log_fn=lambda *_: None)
        names = sorted(p.name for p in Path(d).iterdir() if p.name.startswith("checkpoint-"))
    assert names == ["checkpoint-2", "checkpoint-4"], names


def test_cached_encoder_encodes_each_prompt_once_and_batches():
    from scripts.train_draft import make_cached_encoder
    calls = []

    class FakePipe:
        pass

    import scripts.train_draft as td
    orig = td._encode_prompts_cfg

    def fake_encode(pipe, texts, device):
        calls.append(tuple(texts))
        B = len(texts)
        return (torch.ones(B, 4, 3) * len(texts[0]), torch.zeros(B, 4, 3), torch.ones(B, 2), torch.zeros(B, 2))

    td._encode_prompts_cfg = fake_encode
    try:
        enc = make_cached_encoder(FakePipe(), "cpu")
        a = enc(["aa", "b", "aa"])
        b = enc(["b", "aa"])
    finally:
        td._encode_prompts_cfg = orig
    assert calls == [("aa",), ("b",)], calls          # each distinct prompt encoded once, singly
    assert a[0].shape == (3, 4, 3) and b[0].shape == (2, 4, 3)
    assert float(a[0][0, 0, 0]) == 2.0 and float(a[0][1, 0, 0]) == 1.0 and float(b[0][1, 0, 0]) == 2.0


def test_collapse_guard_stops_on_reward_cliff():
    """Reward that drops off a cliff mid-run must trip the guard, not run to max_steps."""
    import scripts.train_draft as td
    pipe = _make_stub_pipe(batch=2)
    calls = {"n": 0}

    class CliffPredictor:
        def __call__(self, image, prompt_text):
            calls["n"] += 1
            base = image.view(image.shape[0], -1).mean(dim=-1)
            # first 30 calls: fine; afterwards: a constant large penalty (the cliff)
            return base - (0.0 if calls["n"] <= 30 else 1.0)

    with tempfile.TemporaryDirectory() as d:
        s = train_draft(pipe, CliffPredictor(), ["p"], _cfg(max_steps=500, lr=1e-3, collapse_drop=0.3,
                        collapse_window=5, collapse_patience=5), d, device="cpu", encode_fn=_encode, log_fn=lambda *_: None)
    assert s["stop_reason"] == "collapse_guard" and 35 <= s["steps"] <= 60, s


def _log_rewards(d):
    return [(json.loads(l)["step"], json.loads(l)["reward"]) for l in Path(d, "train_log.jsonl").read_text().splitlines()]


def test_resume_matches_uninterrupted():
    """Preempt-and-resume must reproduce the uninterrupted run exactly.

    Run A: 12 steps straight. Run B: 8 steps with an exact-state checkpoint at
    step 6 (so steps 7-8 are 'lost work' logged after the last checkpoint, as
    after a real preemption), then a fresh process with perturbed weights
    resumes from checkpoint-6 to 12. Params, reward log and Adam state must
    match A bitwise; the log must not contain duplicated steps.
    """
    prompts = ["p1", "p2", "p3"]
    torch.manual_seed(5)
    pipe_a = _make_stub_pipe(batch=2)
    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        sa = train_draft(pipe_a, StubPredictor(), prompts, _cfg(max_steps=12), da,
                         device="cpu", encode_fn=_encode, log_fn=lambda *_: None)
        torch.manual_seed(5)
        pipe_b = _make_stub_pipe(batch=2)
        sb1 = train_draft(pipe_b, StubPredictor(), prompts, _cfg(max_steps=8, save_every=6), db,
                          device="cpu", encode_fn=_encode, log_fn=lambda *_: None, resume="none")
        assert sb1["steps"] == 8 and Path(db, "checkpoint-6", "train_state.pt").is_file()
        assert not Path(db, "checkpoint-8", "train_state.pt").exists(), "final save must stay weight-only"
        assert len(_log_rewards(db)) == 8
        # "new process": same non-trainable stubs (same seed), trainable params perturbed
        torch.manual_seed(5)
        pipe_c = _make_stub_pipe(batch=2)
        with torch.no_grad():
            for p_ in pipe_c.transformer.parameters():
                p_.add_(1.0)
        msgs = []
        sc = train_draft(pipe_c, StubPredictor(), prompts, _cfg(max_steps=12), db,
                         device="cpu", encode_fn=_encode, log_fn=msgs.append, resume="auto")
        assert sc["resumed_from"].endswith("checkpoint-6"), sc["resumed_from"]
        assert sc["steps"] == 12 and sc["micro_batches"] == sa["micro_batches"]
        la, lc = _log_rewards(da), _log_rewards(db)
        assert [st for st, _ in lc] == list(range(1, 13)), "log has duplicated/missing steps after resume"
        for (sa_, ra), (sc_, rc) in zip(la, lc):
            assert sa_ == sc_ and abs(ra - rc) < 1e-12, f"step {sa_}: {ra} vs {rc}"
        for (na, pa), (nc, pc) in zip(pipe_a.transformer.named_parameters(), pipe_c.transformer.named_parameters()):
            assert na == nc and torch.equal(pa, pc), f"param {na} differs after resume"
        assert sc["peak_step"] == sa["peak_step"]
        assert any("RESUMED" in m for m in msgs)
        # a changed lr must be refused, not silently resumed under the old name
        torch.manual_seed(5)
        pipe_d = _make_stub_pipe(batch=2)
        try:
            train_draft(pipe_d, StubPredictor(), prompts, _cfg(max_steps=13, lr=5e-3), db,
                        device="cpu", encode_fn=_encode, log_fn=lambda *_: None, resume="auto")
            raise AssertionError("resume with a different lr was not refused")
        except RuntimeError as e:
            assert "config differs" in str(e)
    print(f"  [resume] 12-step run reproduced bitwise from a step-6 checkpoint")


def test_load_prompts_json_and_txt():
    with tempfile.TemporaryDirectory() as d:
        j = Path(d, "p.json"); j.write_text(json.dumps([{"prompt": "a cat"}, "a dog"]))
        t = Path(d, "p.txt"); t.write_text("x\n\ny \n")
        assert load_prompts(j) == ["a cat", "a dog"]
        assert load_prompts(t) == ["x", "y"]


def test_real_prompt_file_loads():
    ps = load_prompts(PROJECT_ROOT / "scripts" / "cloud" / "prompts_smoke.json")
    assert len(ps) == 30 and all(isinstance(p, str) and p for p in ps)


def _run_all():
    tests = [
        test_draft_step_grad_reaches_policy,
        test_loop_learns_and_writes_artifacts,
        test_wall_clock_budget_stops,
        test_grad_accum_counts_micro_batches,
        test_periodic_checkpoints_by_step,
        test_cached_encoder_encodes_each_prompt_once_and_batches,
        test_collapse_guard_stops_on_reward_cliff,
        test_resume_matches_uninterrupted,
        test_load_prompts_json_and_txt,
        test_real_prompt_file_loads,
    ]
    fails = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            fails.append(t.__name__)
    print(f"\n{len(tests) - len(fails)}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_run_all())
