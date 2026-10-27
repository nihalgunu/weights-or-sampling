"""
CPU test for eval_study_matrix.arms_already_done: which arms of a partial
matrix file can be reused after a crash / lost box.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _import():
    # eval_study_matrix imports GPU-only modules at the top; load the function from source instead.
    import ast, types
    src = (PROJECT_ROOT / "scripts" / "cloud" / "eval_study_matrix.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "arms_already_done")
    mod = types.ModuleType("m"); mod.Path = Path
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "eval_study_matrix.py", "exec"), mod.__dict__)
    return mod.arms_already_done


def _scores(n, bases, keys, short=None):
    return {k: {str(b): [0.1] * (short if (short and k == keys[-1] and b == bases[-1]) else n) for b in bases}
            for k in keys}


def test_arms_already_done():
    f = _import()
    keys = ["clip", "pickscore", "jpeg"]
    bases, n = [0, 1000], 3
    meta = {"prompts": ["a", "b", "c"], "seed_bases": bases, "num_steps": 40, "guidance_scale": 4.5,
            "reward": "clip", "base": "sd3", "tfg": {"apply_every_n": 1}}
    arms = [{"name": "no_lora"}, {"name": "tfg_rho20"}, {"name": "draft_s42"}]
    prev = {"meta": dict(meta), "scores": {
        "no_lora": _scores(n, bases, keys),
        "tfg_rho20": _scores(n, bases, keys, short=2),      # incomplete last seed base
    }}
    done = f(prev, arms, bases, n, keys, None, meta)
    assert set(done) == {"no_lora"}, done
    # meta mismatch -> nothing reused
    assert f(prev, arms, bases, n, keys, None, {**meta, "num_steps": 28}) == {}
    assert f(None, arms, bases, n, keys, None, meta) == {}
    # with an image dir, PNGs must all be present for the arm
    with tempfile.TemporaryDirectory() as d:
        assert f(prev, arms, bases, n, keys, d, meta) == {}
        for b in bases:
            p = Path(d, "no_lora", str(b)); p.mkdir(parents=True)
            for i in range(n):
                (p / f"{i:03d}.png").write_bytes(b"x")
        assert set(f(prev, arms, bases, n, keys, d, meta)) == {"no_lora"}
        Path(d, "no_lora", "1000", "002.png").unlink()
        assert f(prev, arms, bases, n, keys, d, meta) == {}


def _run_all():
    fails = []
    for t in [test_arms_already_done]:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}"); traceback.print_exc(); fails.append(t.__name__)
    print(f"\n{1 - len(fails)}/1 passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_run_all())
