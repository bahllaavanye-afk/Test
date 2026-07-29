"""Root principle #5 is enforced on ONE path, and nothing checked it was still wired.

`CLAUDE.md` states: *"Walk-forward only: no in-sample-only backtests are
accepted as valid."* Audited 2026-07-29 — there are **three** separately-written
walk-forward implementations in this repo, and only the first enforces it:

  app/backtest/walk_forward.walk_forward()          LIVE — one call site,
                                                    api/v1/backtests.py:236
  .github/scripts/ml_experiment.walk_forward()      LIVE — local to that script
                                                    (GBM experiments), unrelated
                                                    to the app implementation
  app/ml/training/walk_forward_validate()           DEAD — 125 lines, torch, and
                                                    its ONLY textual reference is
                                                    a string inside its own
                                                    ImportError message

So the nightly ML retrain path — `ml_retrain.retrain_model` -> `train_lstm.train`
— validates on a **single chronological holdout** (`val_frac=0.15`,
`shuffle=False`, contiguous slices), not walk-forward. That is a legitimate
holdout, not a leak: the split is ordered and does not shuffle future into past.
But it is NOT what the root principle says, and the function written to make it
so is dead.

TWO CORRECTIONS to the IMPROVEMENTS entry this audit came from:

  * It claimed `app/backtest/walk_forward.walk_forward()` is called from
    `.github/scripts/ml_experiment.py:152`. **It is not.** That line calls a
    same-named function defined locally at `ml_experiment.py:96`. The app-level
    function has ONE production call site, not two — so the principle rests on a
    single wire.
  * It cited `deflated_sharpe_ratio` at `walk_forward.py:71`; it is at :95.
    Still genuinely wired (`walk_forward.py:9` imports it) — line drift only.

This test does not change behaviour. It pins the state so that:
  1. unwiring the ONE live call site fails loudly, instead of silently ending
     enforcement of a root architectural principle; and
  2. the ML-training gap stays visible rather than being quietly forgotten —
     the same idiom, and the same stated reason, as
     `test_factor_exposure_is_still_honestly_unwired`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[2]
_REPO = _BACKEND.parent

_APP_WF = _BACKEND / "app" / "backtest" / "walk_forward.py"
_ML_WF = _BACKEND / "app" / "ml" / "training" / "walk_forward.py"
_BACKTEST_API = _BACKEND / "app" / "api" / "v1" / "backtests.py"
_TRAIN_LSTM = _BACKEND / "app" / "ml" / "training" / "train_lstm.py"


def _calls_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


# ── the wire that actually enforces the principle ────────────────────────────

def test_the_files_are_where_we_think():
    for p in (_APP_WF, _ML_WF, _BACKTEST_API, _TRAIN_LSTM):
        assert p.is_file(), f"missing {p}"


def _imported_binding(path: Path, module: str, name: str) -> str | None:
    """The LOCAL name `module.name` is bound to, or None if not imported.

    Substring matching is not enough here: `import walk_forward as _x` contains
    the text `import walk_forward`, so a naive `in src` check passes against an
    aliased-away import. Found by testing this guard's own negative case.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == module:
            for a in n.names:
                if a.name == name:
                    return a.asname or a.name
    return None


def test_the_backtest_api_still_calls_walk_forward():
    """The single production call site. If this goes, so does the principle."""
    bound = _imported_binding(_BACKTEST_API, "app.backtest.walk_forward", "walk_forward")
    assert bound is not None, (
        "api/v1/backtests.py no longer imports walk_forward — root CLAUDE.md "
        "principle #5 ('walk-forward only') would no longer be enforced anywhere "
        "in the strategy backtest path"
    )
    assert bound in _calls_in(_BACKTEST_API), (
        f"walk_forward is imported as {bound!r} but never called — importing it "
        f"does not run it"
    )


def test_walk_forward_still_applies_the_overfit_correction():
    """Walk-forward without the multiple-testing correction is half the guard."""
    src = _APP_WF.read_text(encoding="utf-8")
    assert "deflated_sharpe_ratio" in src, (
        "walk_forward no longer computes the Deflated Sharpe Ratio, so best-of-n "
        "selection luck is no longer corrected for"
    )
    assert "deflated_sharpe_ratio" in _calls_in(_APP_WF), "imported but not called"


# ── the gap, pinned honestly ─────────────────────────────────────────────────

def test_ml_walk_forward_validate_is_still_dead():
    """Not a demand that it stay dead — a demand that the record stay true.

    If someone wires it, this fails and the docstring above (and the
    IMPROVEMENTS entry) must be updated to say ML training IS walk-forward
    validated. Silence is what let it sit unused for months.
    """
    refs: list[str] = []
    for path in sorted(_REPO.rglob("*.py")):
        if any(part in {".git", "__pycache__", "tests"} for part in path.parts):
            continue
        try:
            src = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "walk_forward_validate" not in src:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            if "walk_forward_validate" not in line:
                continue
            stripped = line.strip()
            # its own def, and the string inside its own ImportError, are not uses
            if stripped.startswith("def walk_forward_validate"):
                continue
            if "ImportError" in line or "must be installed" in line:
                continue
            refs.append(f"{path.relative_to(_REPO)}:{i}: {stripped[:90]}")

    assert not refs, (
        "walk_forward_validate() now has real call sites:\n  "
        + "\n  ".join(refs)
        + "\n\nThat is good — but update this test's docstring and the "
          "IMPROVEMENTS entry, which both state that ML training is NOT "
          "walk-forward validated."
    )


def test_the_nightly_retrain_still_uses_a_chronological_holdout():
    """Documents WHAT ML training does instead, so 'not walk-forward' is not
    mistaken for 'unvalidated' or for 'leaking future data'."""
    src = _TRAIN_LSTM.read_text(encoding="utf-8")
    assert "val_frac" in src, "train_lstm no longer takes a validation fraction"
    assert "shuffle=False" in src, (
        "train_lstm's loader no longer pins shuffle=False — a shuffled split on "
        "time series leaks future into past, which is a real defect rather than "
        "the documented holdout-vs-walk-forward gap"
    )


def test_there_are_still_three_distinct_implementations():
    """Guards the correction: ml_experiment defines its OWN walk_forward and does
    not import the app one. Conflating them is the error the record contained."""
    exp = _REPO / ".github" / "scripts" / "ml_experiment.py"
    if not exp.is_file():
        pytest.skip("ml_experiment.py not present")
    src = exp.read_text(encoding="utf-8")
    assert "def walk_forward(" in src, "ml_experiment no longer defines its own"
    assert "from app.backtest.walk_forward import" not in src, (
        "ml_experiment now imports the app implementation — the record saying "
        "these are separate is out of date"
    )
