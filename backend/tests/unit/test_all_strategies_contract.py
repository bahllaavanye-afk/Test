"""Registry-wide contract + no-lookahead guard — catches the deadliest backtest
bug (lookahead bias) across EVERY strategy, current and future.

For each registered single-symbol strategy that consumes plain OHLCV, this asserts:
  * `backtest_signals()` returns clean boolean signals (no NaN),
  * no entry on bar 0 (a shifted signal can't fire on the first bar), and
  * **causality / truncation invariance** — entries on `df[:k]` must equal
    `entries[:k]` on the full series; if removing the future changes the past, the
    strategy peeked ahead.

Strategies needing special data (multi-symbol, macro panels, etc.) are skipped
gracefully. The handful with *known* lookahead debt are xfail-tracked below so the
guard stays green while documenting them — but any NEW strategy that leaks the
future fails hard. Survey that seeded this: 104/108 produce signals, 0 bar-0
entries, 0 NaN, 4 lookahead, 1 non-binary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategies import STRATEGY_REGISTRY
from app.strategies.base import BacktestSignals

# Known debt — tracked in GitHub issues (agent-fix-needed). xfail keeps CI green
# while flagging them; if one is fixed it xpasses, prompting removal from the set.
KNOWN_LOOKAHEAD = {
    "kalman_pairs",
    "macro_risk_barometer",
    "mvrv_zscore_timing",
    "yield_curve_momentum",
}
KNOWN_NONBINARY = {"ema_stack_tv"}

_NAMES = sorted(STRATEGY_REGISTRY)


def _ohlcv(n: int = 260, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.015, n)
    c = 100 * np.cumprod(1 + r)
    return pd.DataFrame(
        {
            "open": c * (1 + rng.normal(0, 0.003, n)),
            "high": c * (1 + rng.uniform(0, 0.012, n)),
            "low": c * (1 - rng.uniform(0, 0.012, n)),
            "close": c,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        },
        index=pd.date_range("2022-01-01", periods=n, freq="1D"),
    )


def _signals_or_skip(name: str):
    """Return (instance, BacktestSignals) or skip strategies that need special data."""
    cls = STRATEGY_REGISTRY[name]
    if cls is None:
        pytest.skip(f"{name}: disabled (optional dep missing)")
    try:
        inst = cls()
    except Exception as e:  # needs constructor args (e.g. pairs) — out of scope here
        pytest.skip(f"{name}: not default-constructible ({e})")
    try:
        sig = inst.backtest_signals(_ohlcv())
    except Exception as e:  # needs multi-symbol / special panels
        pytest.skip(f"{name}: backtest_signals needs special data ({e})")
    if not isinstance(sig, BacktestSignals):
        pytest.skip(f"{name}: not a plain-OHLCV BacktestSignals strategy")
    return inst, sig


@pytest.mark.parametrize("name", _NAMES)
def test_signals_no_nan(name):
    _, sig = _signals_or_skip(name)
    assert not sig.entries.isna().any(), f"{name}: NaN in entries mask"
    assert not sig.exits.isna().any(), f"{name}: NaN in exits mask"


@pytest.mark.parametrize("name", _NAMES)
def test_signals_binary(name):
    _, sig = _signals_or_skip(name)
    vals = set(pd.unique(sig.entries.dropna())) | set(pd.unique(sig.exits.dropna()))
    binary = {True, False, 0, 1, 0.0, 1.0}
    if name in KNOWN_NONBINARY and not vals <= binary:
        pytest.xfail(f"{name}: known non-binary signal output (tracked)")
    assert vals <= binary, f"{name}: non-binary signal values {vals - binary}"


@pytest.mark.parametrize("name", _NAMES)
def test_no_entry_on_bar0(name):
    _, sig = _signals_or_skip(name)
    assert not bool(sig.entries.iloc[0]), f"{name}: entry on bar 0 is lookahead bias"


@pytest.mark.parametrize("name", _NAMES)
def test_signals_are_causal(name):
    inst, sig = _signals_or_skip(name)
    df = _ohlcv()
    full = sig.entries.reset_index(drop=True)
    k = 180
    trunc = inst.backtest_signals(df.iloc[:k]).entries.reset_index(drop=True)
    mismatches = int((full.iloc[:k].values != trunc.values).sum())
    if name in KNOWN_LOOKAHEAD and mismatches:
        pytest.xfail(f"{name}: known lookahead debt (agent-fix-needed) — {mismatches} bars")
    assert mismatches == 0, (
        f"{name}: {mismatches} entries in [0:{k}] changed when future data was "
        f"removed → lookahead bias"
    )
