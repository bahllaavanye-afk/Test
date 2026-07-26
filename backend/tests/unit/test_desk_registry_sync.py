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

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
ENV_SECRET_KEY = "SECRET_KEY"
ENV_SECRET_KEY_DEFAULT = "test-secret-key-32-bytes-hex-xxxxxx"
ENV_DATABASE_URL = "DATABASE_URL"
ENV_DATABASE_URL_DEFAULT = "sqlite+aiosqlite:///./test_desk_sync.db"

DESK_MODULE_REL_PATH = Path(".github") / "scripts" / "desk_order_placer.py"
DESK_MODULE_SKIP_MSG = f"{DESK_MODULE_REL_PATH.name} not present in this checkout"
DESK_MODULE_SPEC_NAME = "dop_sync_test"

FX_DESK_FILENAME = "fx_desk.py"
FX_DESK_SKIP_MSG = f"{FX_DESK_FILENAME} not present"
FX_SPEC_NAME = "fx_sync_test"

ASSERT_DESK_MSG_TEMPLATE = "desks reference unknown/unloadable strategies: {missing}"
ASSERT_FX_MSG_TEMPLATE = "FX desk references unknown strategies: {missing}"

# ----------------------------------------------------------------------
# Environment setup
# ----------------------------------------------------------------------
os.environ.setdefault(ENV_SECRET_KEY, ENV_SECRET_KEY_DEFAULT)
os.environ.setdefault(ENV_DATABASE_URL, ENV_DATABASE_URL_DEFAULT)

_DESK_MOD = Path(__file__).parents[3] / DESK_MODULE_REL_PATH


def _load_desks():
    if not _DESK_MOD.exists():
        pytest.skip(DESK_MODULE_SKIP_MSG)
    spec = importlib.util.spec_from_file_location(DESK_MODULE_SPEC_NAME, _DESK_MOD)
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
    assert not missing, ASSERT_DESK_MSG_TEMPLATE.format(missing=sorted(missing))


def test_fx_desk_strategies_exist_in_registry():
    from app.strategies import STRATEGY_REGISTRY

    fx_mod = _DESK_MOD.parent / FX_DESK_FILENAME
    if not fx_mod.exists():
        pytest.skip(FX_DESK_SKIP_MSG)
    spec = importlib.util.spec_from_file_location(FX_SPEC_NAME, fx_mod)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    missing = [s for s in m.STRATEGIES if STRATEGY_REGISTRY.get(s) is None]
    assert not missing, ASSERT_FX_MSG_TEMPLATE.format(missing=missing)