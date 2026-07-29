"""Synthetic options backtester: pricing correctness + structure behavior."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.backtest.options_synthetic import (
    BULL_PUT_SPREAD,
    IRON_CONDOR,
    SpreadLeg,
    backtest_spread,
    bs_price,
    price_spread,
    realized_vol,
)


# ── Black-Scholes correctness ─────────────────────────────────────────────────

def test_put_call_parity():
    S, K, t, sig, r = 100.0, 100.0, 0.25, 0.2, 0.04
    c = bs_price(S, K, t, sig, "call", r)
    p = bs_price(S, K, t, sig, "put", r)
    assert c - p == pytest.approx(S - K * math.exp(-r * t), abs=1e-9)


def test_expiry_returns_intrinsic():
    assert bs_price(110, 100, 0.0, 0.2, "call") == pytest.approx(10.0)
    assert bs_price(90, 100, 0.0, 0.2, "put") == pytest.approx(10.0)
    assert bs_price(110, 100, 0.25, 0.0, "put") == pytest.approx(0.0)


def test_price_monotone_in_vol_and_positive():
    lo = bs_price(100, 105, 0.1, 0.10, "call")
    hi = bs_price(100, 105, 0.1, 0.40, "call")
    assert 0 <= lo < hi


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        bs_price(0, 100, 0.1, 0.2, "call")


def test_bs_price_none_inputs_raise():
    """Ensure None values are rejected rather than silently processed."""
    with pytest.raises(ValueError):
        bs_price(None, 100, 0.1, 0.2, "call")
    with pytest.raises(ValueError):
        bs_price(100, None, 0.1, 0.2, "call")
    with pytest.raises(ValueError):
        bs_price(100, 100, None, 0.2, "call")
    with pytest.raises(ValueError):
        bs_price(100, 100, 0.1, None, "call")
    with pytest.raises(ValueError):
        bs_price(100, 100, 0.1, 0.2, None)  # type: ignore[arg-type]


# ── Spread pricing ────────────────────────────────────────────────────────────

def test_iron_condor_enters_at_net_credit():
    strikes = [m * 100 for m in (0.95, 0.91, 1.05, 1.09)]
    v = price_spread(100.0, IRON_CONDOR, strikes, 35 / 252, 0.25)
    assert v < 0            # holder is net short premium


def test_bull_put_spread_defined_risk():
    strikes = [96.0, 92.0]
    entry = price_spread(100.0, BULL_PUT_SPREAD, strikes, 35 / 252, 0.25)
    crash = price_spread(60.0, BULL_PUT_SPREAD, strikes, 0.0, 0.25)
    assert entry < 0                                # credit at entry
    assert crash == pytest.approx(-(96 - 92))       # loss capped at width
    assert (crash - entry) == pytest.approx(-(96 - 92) - entry)


def test_price_spread_empty_strikes_raise():
    """Empty strike list should raise a clear error."""
    with pytest.raises(ValueError):
        price_spread(100.0, IRON_CONDOR, [], 35 / 252, 0.25)


def test_price_spread_mismatched_strikes_raise():
    """Providing an incorrect number of strikes for a spread should raise."""
    # BULL_PUT_SPREAD expects exactly two strikes
    with pytest.raises(ValueError):
        price_spread(100.0, BULL_PUT_SPREAD, [95.0], 35 / 252, 0.25)
    with pytest.raises(ValueError):
        price_spread(100.0, BULL_PUT_SPREAD, [95.0, 90.0, 85.0], 35 / 252, 0.25)


# ── Backtest behavior ─────────────────────────────────────────────────────────

def _flat_df(n=250, price=100.0, noise=0.002, seed=3):
    rs = np.random.RandomState(seed)
    close = price * np.exp(np.cumsum(rs.normal(0, noise, n)))
    return pd.DataFrame(
        {
            "close": close,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "volume": [1e6] * n,
        },
        index=pd.date_range("2025-01-01", periods=n, freq="B"),
    )


def test_short_premium_profits_in_flat_tape():
    res = backtest_spread(_flat_df(), IRON_CONDOR, dte=35, hold_days=21)
    assert res.trades >= 20
    assert res.total_pnl > 0                        # theta decay harvested
    assert res.win_rate and res.win_rate > 0.6


def test_short_premium_bleeds_in_crash():
    n = 250
    close = np.concatenate(
        [
            np.full(n // 2, 100.0),
            100.0 * np.exp(np.linspace(0, -0.5, n - n // 2)),
        ]
    )
    df = pd.DataFrame(
        {
            "close": close,
            "open": close,
            "high": close,
            "low": close,
            "volume": [1e6] * n,
        },
        index=pd.date_range("2025-01-01", periods=n, freq="B"),
    )
    res = backtest_spread(df, BULL_PUT_SPREAD, dte=35, hold_days=21)
    assert res.max_loss < 0                         # crash regime hurts short puts


def test_realized_vol_shape_and_result_summary():
    df = _flat_df()
    rv = realized_vol(df["close"])
    assert rv.iloc[25] > 0
    res = backtest_spread(df, [SpreadLeg("call", "buy", 1.0)], dte=35, hold_days=21)
    assert "trades" in res.summary and res.trades > 0


def test_realized_vol_empty_series_raise():
    """An empty price series should raise an informative error."""
    empty_series = pd.Series(dtype=float)
    with pytest.raises(ValueError):
        realized_vol(empty_series)


def test_backtest_spread_none_dataframe_raise():
    """Passing None as the price DataFrame must raise."""
    with pytest.raises(ValueError):
        backtest_spread(None, IRON_CONDOR, dte=35, hold_days=21)  # type: ignore[arg-type]