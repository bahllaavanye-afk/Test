"""Guards for the continuous improver's safety rails.

Two invariants that stop the improver from re-creating the stuck/harmful PRs:
  1. Core trading-logic modules (strategies/execution/risk/ml.models/bots) are
     OFF-LIMITS — the improver does whole-file LLM rewrites with no per-change
     behavior test, so a green PR can still regress the money path.
  2. `pick_target_file` never returns a protected file, on either the primary
     glob or the fallback-to-all-files path.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import continuous_improver as ci


def test_is_protected_flags_core_trading_modules():
    assert ci._is_protected("backend/app/strategies/manual/mean_reversion.py")
    assert ci._is_protected("backend/app/strategies/ml_enhanced/ml_breakout.py")
    assert ci._is_protected("backend/app/execution/vwap.py")
    assert ci._is_protected("backend/app/risk/circuit_breaker.py")
    assert ci._is_protected("backend/app/ml/models/lstm.py")
    assert ci._is_protected("backend/app/bots/engine.py")


def test_is_protected_allows_non_core_modules():
    assert not ci._is_protected("backend/app/api/v1/trades.py")
    assert not ci._is_protected("backend/app/schemas/bot.py")
    assert not ci._is_protected("backend/app/utils/logging.py")
    assert not ci._is_protected("backend/app/tasks/agent_memory.py")
    # ml features (not models) are not in the protected money-path set
    assert not ci._is_protected("backend/app/ml/features/engineer.py")


def test_pick_target_file_never_returns_protected(monkeypatch):
    # `_too_large` stats the path and returns True on OSError, so these fake
    # relative paths silently become "unusable" unless pytest happens to be run
    # from the repo root — the whole list empties and `pick_target_file` returns
    # None, failing on an assertion that has nothing to do with what is tested
    # here. Neutralise the size guard explicitly; protected-path filtering is
    # the subject, and `test_oversized_files_are_never_picked` covers the guard.
    monkeypatch.setattr(ci, "_too_large", lambda p: False)
    mixed = [
        "backend/app/strategies/manual/mean_reversion.py",  # protected
        "backend/app/execution/iceberg.py",                 # protected
        "backend/app/risk/hrp.py",                          # protected
        "backend/app/api/v1/trades.py",                     # ok
        "backend/app/schemas/bot.py",                       # ok
    ]
    monkeypatch.setattr(ci.glob, "glob", lambda *a, **k: list(mixed))
    for hour in range(24):
        for _ in range(20):
            picked = ci.pick_target_file(hour, set())
            assert picked is not None
            assert not ci._is_protected(picked), f"picked protected file: {picked}"


def test_pick_target_file_fallback_also_filters_protected(monkeypatch):
    # Primary pattern yields only protected files → falls through to the
    # all-files branch, which must ALSO exclude protected paths.
    monkeypatch.setattr(ci, "_too_large", lambda p: False)  # see the note above
    calls = {"n": 0}

    def fake_glob(pattern, *a, **k):
        calls["n"] += 1
        if "**" in pattern:  # the fallback all-files glob
            return ["backend/app/risk/hrp.py", "backend/app/utils/logging.py"]
        return ["backend/app/strategies/manual/mean_reversion.py"]  # primary: protected only

    monkeypatch.setattr(ci.glob, "glob", fake_glob)
    picked = ci.pick_target_file(3, set())
    assert picked == "backend/app/utils/logging.py"
    assert calls["n"] >= 2  # primary emptied by filter → fallback consulted


def test_oversized_files_are_never_picked(monkeypatch, tmp_path):
    """The #1248 guard itself, which had no test.

    Half of one run's ten attempts went to files `improve_file()` rejects on
    sight for exceeding MAX_FILE_CHARS. The filter does not change which files
    are improvable — those were already rejected 100% of the time — it stops
    the budget being spent on a guaranteed rejection. Real files on disk here,
    because the guard's whole job is to stat them.
    """
    small = tmp_path / "small.py"
    small.write_text("x = 1\n")
    big = tmp_path / "big.py"
    big.write_text("# pad\n" * ci.MAX_FILE_CHARS)
    assert big.stat().st_size > ci.MAX_FILE_CHARS

    monkeypatch.setattr(ci.glob, "glob", lambda *a, **k: [str(big), str(small)])
    for _ in range(30):
        assert ci.pick_target_file(0, set()) == str(small)


def test_an_unreadable_path_is_treated_as_unusable():
    """OSError → True, deliberately. A path that cannot be stat'd cannot be
    read by `improve_file()` either, so picking it would burn an attempt."""
    assert ci._too_large("backend/app/definitely/not/here.py") is True
