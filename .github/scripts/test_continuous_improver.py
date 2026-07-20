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
