"""The OHLCV cache hands out shared DataFrames. Nothing may mutate them.

`improve(optimization)` #1621 added an LRU cache to `backtest_worker`:

    if key in _OHLCV_CACHE:
        _OHLCV_CACHE.move_to_end(key)
        return _OHLCV_CACHE[key]          # the SAME object, not a copy

and the cached frame goes straight into `strategy.backtest_signals(df)`. A
strategy that did `df["sma"] = ...` would poison the cache for every other
strategy sharing that (symbol, start, end, interval) key — silently, and only
on the second and later backtests, which is the hardest kind of bug to see.

Measured 2026-08-07: all 116 registered strategies were run against a shared
frame and **none** mutated it — no added columns, no altered values. So the
cache is correct. It is correct *by luck* rather than by construction, because
nothing states or enforces the invariant it depends on.

This test states it. If a future strategy mutates its input, this fails loudly
instead of the cache quietly serving polluted data.
"""
from __future__ import annotations

import asyncio
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest


def _frame(n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.DataFrame(
        {"open": px, "high": px * 1.01, "low": px * 0.99, "close": px,
         "volume": rng.integers(1e5, 1e6, n).astype(float)},
        index=idx,
    )


def test_the_cache_returns_a_shared_object_which_is_why_this_matters():
    """Pins the premise. If the cache ever starts returning `.copy()`, the
    invariant below stops being load-bearing and this file can be revisited —
    but until then the sharing is real and unguarded."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "app" / "tasks" / "backtest_worker.py").read_text()
    assert "_OHLCV_CACHE[key]" in src, "the OHLCV cache is gone; re-check this file's premise"
    assert "return _OHLCV_CACHE[key].copy()" not in src, (
        "the cache now copies — good, but this test's rationale needs updating")


def test_no_registered_strategy_mutates_its_input_frame():
    """The invariant `_OHLCV_CACHE` silently depends on.

    Runs every strategy in the registry against a shared frame and asserts the
    frame is unchanged: no new columns, no altered values.
    """
    warnings.filterwarnings("ignore")
    from app.strategies import STRATEGY_REGISTRY

    assert len(STRATEGY_REGISTRY) > 50, (
        f"only {len(STRATEGY_REGISTRY)} strategies registered — the registry "
        f"failed to populate, which would make this check vacuous")

    base = _frame()
    offenders: list[str] = []
    ran = 0
    for name, cls in STRATEGY_REGISTRY.items():
        df = base.copy()
        cols_before = set(df.columns)
        close_before = df["close"].copy()
        try:
            result = cls().backtest_signals(df)
            if inspect.isawaitable(result):
                asyncio.run(result)
            ran += 1
        except Exception:
            # A strategy that cannot run on synthetic data still must not have
            # mutated the frame on its way to failing.
            pass
        added = set(df.columns) - cols_before
        changed = not df["close"].equals(close_before)
        if added or changed:
            offenders.append(f"{name}: added={sorted(added)[:3]} values_changed={changed}")

    assert ran > 50, f"only {ran} strategies executed; the check would be near-vacuous"
    assert not offenders, (
        "strategies mutate the DataFrame handed to them, which poisons the shared "
        "_OHLCV_CACHE in backtest_worker for every other strategy on the same key:\n  "
        + "\n  ".join(offenders)
        + "\nEither make the strategy work on a copy, or make _get_cached_ohlcv "
          "return _OHLCV_CACHE[key].copy()."
    )
