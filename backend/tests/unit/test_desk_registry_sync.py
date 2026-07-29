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


def _load_module_from_path(file_path: Path, module_name: str):
    """Load a Python module from the given file path, skipping the test if missing."""
    if not file_path.exists():
        pytest.skip(f"{file_path.name} not present")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Unable to create module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_desks():
    """Load the desk definitions from the GitHub Actions script."""
    return _load_module_from_path(_DESK_MOD, "dop_sync_test")


def _collect_missing_strategies(desks, registry) -> set[str]:
    """Return a set of missing strategy identifiers in the form 'desk_name/strategy'."""
    missing: set[str] = set()
    for desk in desks:
        for strat_name in desk.strategy_names:
            if registry.get(strat_name) is None:
                missing.add(f"{desk.name}/{strat_name}")
    return missing


def _collect_missing_fx_strategies(strategies, registry) -> list[str]:
    """Return a list of FX desk strategy names that are not present in the registry."""
    return [s for s in strategies if registry.get(s) is None]


def test_every_desk_strategy_exists_in_registry():
    from app.strategies import STRATEGY_REGISTRY

    dop = _load_desks()
    missing = _collect_missing_strategies(dop.DESKS, STRATEGY_REGISTRY)
    assert not missing, f"desks reference unknown/unloadable strategies: {sorted(missing)}"


def test_fx_desk_strategies_exist_in_registry():
    from app.strategies import STRATEGY_REGISTRY

    fx_mod_path = _DESK_MOD.parent / "fx_desk.py"
    fx_mod = _load_module_from_path(fx_mod_path, "fx_sync_test")
    missing = _collect_missing_fx_strategies(fx_mod.STRATEGIES, STRATEGY_REGISTRY)
    assert not missing, f"FX desk references unknown strategies: {missing}"