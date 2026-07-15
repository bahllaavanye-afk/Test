"""STRATEGY_REGISTRY must import even when optional deps are missing.

A module-level `import aiohttp` in 3 strategies crashed the whole registry
import when aiohttp wasn't installed — which killed every desk run for days
(ModuleNotFoundError → desk_order_placer FATAL, zero trades). Optional deps
must be lazy; this guards against the regression."""
from __future__ import annotations

import builtins
import importlib
import sys
from typing import Callable

from pydantic import BaseModel, Field, validator


class ImportBlockConfig(BaseModel):
    """Configuration for the import-blocking test.

    Attributes
    ----------
    module_name: str
        Name of the module to block import for. Example: ``"aiohttp"``.
    min_registry_size: int
        Minimum expected size of the strategy registry after import. Example: ``50``.
    """

    module_name: str = Field(..., description="Name of the module to block import for", example="aiohttp")
    min_registry_size: int = Field(..., description="Minimum expected size of the strategy registry", example=50)

    @validator("module_name")
    def module_name_must_not_be_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("module_name must not be empty")
        return v

    @validator("min_registry_size")
    def min_registry_size_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("min_registry_size must be a positive integer")
        return v


def test_registry_imports_without_aiohttp() -> None:
    """Ensure the strategy registry can be imported when ``aiohttp`` is missing.

    The test temporarily blocks imports of ``aiohttp`` (and submodules) and
    verifies that the registry still loads and contains a reasonable number of
    entries.
    """
    config = ImportBlockConfig(module_name="aiohttp", min_registry_size=50)

    real_import: Callable = builtins.__import__

    def blocked(name: str, *args, **kwargs):
        if name == config.module_name or name.startswith(f"{config.module_name}."):
            raise ModuleNotFoundError(f"No module named '{config.module_name}'")
        return real_import(name, *args, **kwargs)

    # Drop any cached strategy modules so the import actually re-runs
    for mod in [m for m in sys.modules if m.startswith("app.strategies")]:
        del sys.modules[mod]

    builtins.__import__ = blocked
    try:
        reg = importlib.import_module("app.strategies").STRATEGY_REGISTRY
        assert len(reg) > config.min_registry_size, f"registry too small: {len(reg)}"
    finally:
        builtins.__import__ = real_import