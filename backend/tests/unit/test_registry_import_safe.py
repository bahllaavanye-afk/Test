"""STRATEGY_REGISTRY must import even when optional deps are missing.

A module-level `import aiohttp` in 3 strategies crashed the whole registry
import when aiohttp wasn't installed — which killed every desk run for days
(ModuleNotFoundError → desk_order_placer FATAL, zero trades). Optional deps
must be lazy; this guards against the regression."""
from __future__ import annotations

import builtins
import importlib
import sys

from pydantic import BaseModel, Field, validator


class RegistryImportTestResult(BaseModel):
    """Result of the strategy registry import test.

    Attributes
    ----------
    registry_length: int
        Number of strategies successfully registered. Must be a positive integer.
        Example: 120
    success: bool
        Indicator that the import completed without raising an exception.
        Example: True
    """
    registry_length: int = Field(..., description="Number of strategies registered", example=120)
    success: bool = Field(..., description="Whether the import succeeded", example=True)

    @validator("registry_length")
    def check_positive_length(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("registry_length must be a positive integer")
        return v


def test_registry_imports_without_aiohttp():
    """Ensures the strategy registry can be imported when `aiohttp` is absent.

    The test temporarily blocks importing the `aiohttp` package and verifies that
    the `STRATEGY_REGISTRY` still loads with a sufficient number of entries.
    """
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "aiohttp" or name.startswith("aiohttp."):
            raise ModuleNotFoundError("No module named 'aiohttp'")
        return real_import(name, *a, **k)

    # Drop any cached strategy modules so the import actually re-runs
    for mod in [m for m in sys.modules if m.startswith("app.strategies")]:
        del sys.modules[mod]
    builtins.__import__ = blocked
    try:
        reg = importlib.import_module("app.strategies").STRATEGY_REGISTRY
        assert len(reg) > 50, f"registry too small: {len(reg)}"
        # Capture test result using the Pydantic schema
        result = RegistryImportTestResult(registry_length=len(reg), success=True)
    finally:
        builtins.__import__ = real_import