"""STRATEGY_REGISTRY must import even when optional deps are missing.

A module-level `import aiohttp` in 3 strategies crashed the whole registry
import when aiohttp wasn't installed — which killed every desk run for days
(ModuleNotFoundError → desk_order_placer FATAL, zero trades). Optional deps
must be lazy; this guards against the regression."""
from __future__ import annotations

import builtins
import importlib
import sys


def test_registry_imports_without_aiohttp():
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "aiohttp" or name.startswith("aiohttp."):
            raise ModuleNotFoundError("No module named 'aiohttp'")
        return real_import(name, *a, **k)

    # drop any cached strategy modules so the import actually re-runs
    for mod in [m for m in sys.modules if m.startswith("app.strategies")]:
        del sys.modules[mod]
    builtins.__import__ = blocked
    try:
        reg = importlib.import_module("app.strategies").STRATEGY_REGISTRY
        assert len(reg) > 50, f"registry too small: {len(reg)}"
    finally:
        builtins.__import__ = real_import
