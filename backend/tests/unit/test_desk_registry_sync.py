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

# Constants
ENV_SECRET_KEY = "SECRET_KEY"
DEFAULT_SECRET_KEY = "test-secret-key-32-bytes-hex-xxxxxx"
ENV_DATABASE_URL = "DATABASE_URL"
DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./test_desk_sync.db"
DESK_ORDER_PLACER_FILENAME = "desk_order_placer.py"
DESK_ORDER_PLACER_SKIP_MSG = f"{DESK_ORDER_PLACER_FILENAME} not present in this checkout"
FX_DESK_FILENAME = "fx_desk.py"
FX_DESK_SKIP_MSG = f"{FX_DESK_FILENAME} not present"
DOP_MODULE_NAME = "dop_sync_test"
FX_MODULE_NAME = "fx_sync_test"
DESK_STRATEGY_MISSING_MSG = "desks reference unknown/unloadable strategies: {}"
FX_STRATEGY_MISSING_MSG = "FX desk references unknown strategies: {}"

os.environ.setdefault(ENV_SECRET_KEY, DEFAULT_SECRET_KEY)
os.environ.setdefault(ENV_DATABASE_URL, DEFAULT_DATABASE_URL)

_DESK_MOD = Path(__file__).parents[3] / ".github" / "scripts" / DESK_ORDER_PLACER_FILENAME


def _load_desks():
    if not _DESK_MOD.exists():
        pytest.skip(DESK_ORDER_PLACER_SKIP_MSG)
    spec = importlib.util.spec_from_file_location(DOP_MODULE_NAME, _DESK_MOD)
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
    assert not missing, DESK_STRATEGY_MISSING_MSG.format(sorted(missing))


def test_fx_desk_strategies_exist_in_registry():
    from app.strategies import STRATEGY_REGISTRY

    fx_mod = _DESK_MOD.parent / FX_DESK_FILENAME
    if not fx_mod.exists():
        pytest.skip(FX_DESK_SKIP_MSG)
    spec = importlib.util.spec_from_file_location(FX_MODULE_NAME, fx_mod)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    missing = [s for s in m.STRATEGIES if STRATEGY_REGISTRY.get(s) is None]
    assert not missing, FX_STRATEGY_MISSING_MSG.format(missing)