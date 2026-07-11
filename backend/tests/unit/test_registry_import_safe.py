"""STRATEGY_REGISTRY must import even when optional deps are missing.

A module-level `import aiohttp` in 3 strategies crashed the whole registry
import when aiohttp wasn't installed — which killed every desk run for days
(ModuleNotFoundError → desk_order_placer FATAL, zero trades). Optional deps
must be lazy; this guards against the regression."""
from __future__ import annotations

import builtins
import importlib
import sys

# Constants
AIOHTTP_MODULE = "aiohttp"
AIOHTTP_NOT_FOUND_MSG = "No module named 'aiohttp'"
STRATEGY_MODULE_PREFIX = "app.strategies"
REGISTRY_ATTR = "STRATEGY_REGISTRY"
MIN_REGISTRY_SIZE = 50
REGISTRY_TOO_SMALL_MSG = "registry too small: {}"


def test_registry_imports_without_aiohttp():
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == AIOHTTP_MODULE or name.startswith(f"{AIOHTTP_MODULE}."):
            raise ModuleNotFoundError(AIOHTTP_NOT_FOUND_MSG)
        return real_import(name, *a, **k)

    # drop any cached strategy modules so the import actually re-runs
    for mod in [m for m in sys.modules if m.startswith(STRATEGY_MODULE_PREFIX)]:
        del sys.modules[mod]
    builtins.__import__ = blocked
    try:
        reg = importlib.import_module(STRATEGY_MODULE_PREFIX).__dict__[REGISTRY_ATTR]
        assert len(reg) > MIN_REGISTRY_SIZE, REGISTRY_TOO_SMALL_MSG.format(len(reg))
    finally:
        builtins.__import__ = real_import