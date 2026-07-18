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
    """Load the desk definition module safely.

    Returns:
        The loaded module, or None if the module cannot be loaded.
    """
    if not _DESK_MOD.exists():
        pytest.skip("desk_order_placer.py not present in this checkout")
    spec = importlib.util.spec_from_file_location("dop_sync_test", _DESK_MOD)
    if spec is None or spec.loader is None:
        # Unable to create a module spec; treat as missing.
        pytest.skip("Unable to load desk_order_placer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def test_every_desk_strategy_exists_in_registry():
    from app.strategies import STRATEGY_REGISTRY

    dop = _load_desks()
    # Guard against None or missing attributes.
    desks = getattr(dop, "DESKS", []) or []
    missing = {
        f"{getattr(d, 'name', '<unknown>')}/{s}"
        for d in desks
        for s in getattr(d, "strategy_names", []) or []
        if STRATEGY_REGISTRY.get(s) is None
    }
    assert not missing, f"desks reference unknown/unloadable strategies: {sorted(missing)}"


def test_fx_desk_strategies_exist_in_registry():
    from app.strategies import STRATEGY_REGISTRY

    fx_mod = _DESK_MOD.parent / "fx_desk.py"
    if not fx_mod.exists():
        pytest.skip("fx_desk.py not present")
    spec = importlib.util.spec_from_file_location("fx_sync_test", fx_mod)
    if spec is None or spec.loader is None:
        pytest.skip("Unable to load fx_desk.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    strategies = getattr(m, "STRATEGIES", []) or []
    missing = [s for s in strategies if STRATEGY_REGISTRY.get(s) is None]
    assert not missing, f"FX desk references unknown strategies: {missing}"