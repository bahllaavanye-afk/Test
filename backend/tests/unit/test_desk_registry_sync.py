"""Desk ↔ registry cross-check, run where the FULL backend deps exist.

The GitHub-Actions desk layer (.github/scripts/desk_order_placer.py) names
backend strategies by string; a typo'd name silently shrinks a desk (the
loader skips unknown names). The lightweight test-agents CI job can't import
the full registry (statsmodels etc.), so THIS test — in the backend job —
carries the authoritative check. test_desk_config.py keeps the dep-free
config checks in the agents job.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-32-bytes-hex-xxxxxx")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_desk_sync.db")

_DESK_MOD = Path(__file__).parents[3] / ".github" / "scripts" / "desk_order_placer.py"


def _load_desks():
    if not _DESK_MOD.exists():
        pytest.skip("desk_order_placer.py not present in this checkout")
    spec = importlib.util.spec_from_file_location("dop_sync_test", _DESK_MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def test_every_desk_strategy_exists_in_registry():
    from app.strategies import STRATEGY_REGISTRY

    dop = _load_desks()
    missing = {
        f"{d.name}/{s}"
        for d in dop.DESKS
        for s in d.strategy_names
        if STRATEGY_REGISTRY.get(s) is None
    }
    assert not missing, f"desks reference unknown/unloadable strategies: {sorted(missing)}"


def test_fx_desk_strategies_exist_in_registry():
    from app.strategies import STRATEGY_REGISTRY

    fx_mod = _DESK_MOD.parent / "fx_desk.py"
    if not fx_mod.exists():
        pytest.skip("fx_desk.py not present")
    spec = importlib.util.spec_from_file_location("fx_sync_test", fx_mod)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    missing = [s for s in m.STRATEGIES if STRATEGY_REGISTRY.get(s) is None]
    assert not missing, f"FX desk references unknown strategies: {missing}"


def test_empty_desks_list(monkeypatch):
    """Edge case: the DESKS collection is empty – should produce no missing entries."""
    from app.strategies import STRATEGY_REGISTRY

    dop = _load_desks()
    # Replace DESKS with an empty list to simulate no desks defined.
    monkeypatch.setattr(dop, "DESKS", [], raising=False)

    missing = {
        f"{d.name}/{s}"
        for d in dop.DESKS
        for s in getattr(d, "strategy_names", [])
        if STRATEGY_REGISTRY.get(s) is None
    }
    assert missing == set(), "Empty DESKS list should result in no missing strategies"


def test_desk_with_unknown_strategy(monkeypatch):
    """Edge case: a desk references a strategy not present in the registry – should be detected."""
    from app.strategies import STRATEGY_REGISTRY

    dop = _load_desks()

    class FakeDesk:
        name = "fake_desk"
        strategy_names = ["non_existent_strategy"]

    monkeypatch.setattr(dop, "DESKS", [FakeDesk()], raising=False)

    missing = {
        f"{d.name}/{s}"
        for d in dop.DESKS
        for s in d.strategy_names
        if STRATEGY_REGISTRY.get(s) is None
    }
    # The missing set should contain the exact unknown strategy reference.
    assert missing == {"fake_desk/non_existent_strategy"}


def test_desk_with_duplicate_strategies(monkeypatch):
    """Edge case: a desk lists the same strategy multiple times – duplicates should not cause false positives."""
    from app.strategies import STRATEGY_REGISTRY

    dop = _load_desks()

    # Choose a known strategy from the registry for the duplicate test.
    known_strategy = next(iter(STRATEGY_REGISTRY.keys()))

    class DuplicateDesk:
        name = "dup_desk"
        strategy_names = [known_strategy, known_strategy]  # duplicate entries

    monkeypatch.setattr(dop, "DESKS", [DuplicateDesk()], raising=False)

    missing = {
        f"{d.name}/{s}"
        for d in dop.DESKS
        for s in d.strategy_names
        if STRATEGY_REGISTRY.get(s) is None
    }
    # Since the strategy is known, missing should be empty despite duplicates.
    assert missing == set()