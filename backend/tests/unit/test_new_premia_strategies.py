"""Behavior tests for the 2026-07-18 premia strategies — signals must fire
under the documented conditions and refuse otherwise (not just not-crash)."""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from app.strategies.manual.double_seven import DoubleSevenStrategy
from app.strategies.manual.gap_fill_fade import GapFillFadeStrategy
from app.strategies.manual.turn_of_month import TurnOfMonthStrategy, _in_tom_window


def _df(closes, opens=None, start="2025-01-01", freq="B"):
    closes = np.asarray(closes, dtype=float)
    opens = np.asarray(opens, dtype=float) if opens is not None else closes.copy()
    idx = pd.bdate_range(start, periods=len(closes)) if freq == "B" else \
        pd.date_range(start, periods=len(closes), freq=freq)
    return pd.DataFrame({"open": opens, "close": closes,
                         "high": np.maximum(opens, closes) * 1.001,
                         "low": np.minimum(opens, closes) * 0.999,
                         "volume": [1e6] * len(closes)}, index=idx)


# ── turn_of_month ─────────────────────────────────────────────────────────────

def test_tom_window_marks_month_boundary():
    idx = pd.bdate_range("2025-01-20", periods=15)          # spans Jan→Feb
    w = _in_tom_window(idx)
    last_jan = max(i for i, d in enumerate(idx) if d.month == 1)
    first_feb = min(i for i, d in enumerate(idx) if d.month == 2)
    assert bool(w.iloc[last_jan]) and bool(w.iloc[first_feb])
    assert bool(w.iloc[first_feb + 2]) and not bool(w.iloc[first_feb + 3])


def test_tom_fires_only_in_window():
    closes = np.linspace(100, 105, 60)
    # index ending on the first trading day of a month → in window
    idx = pd.bdate_range(end="2025-07-01", periods=60)
    df = pd.DataFrame({"open": closes, "close": closes, "high": closes,
                       "low": closes, "volume": [1e6] * 60}, index=idx)
    sig = asyncio.run(TurnOfMonthStrategy().analyze(df, "SPY"))
    assert sig is not None and sig.side == "buy" and 0.6 <= sig.confidence <= 0.8
    # mid-month → None
    idx2 = pd.bdate_range(end="2025-07-16", periods=60)
    df2 = pd.DataFrame({"open": closes, "close": closes, "high": closes,
                        "low": closes, "volume": [1e6] * 60}, index=idx2)
    assert asyncio.run(TurnOfMonthStrategy().analyze(df2, "SPY")) is None


# ── gap_fill_fade ─────────────────────────────────────────────────────────────

def test_gap_fade_buys_down_gap_sells_up_gap():
    closes = np.full(40, 100.0)
    opens = closes.copy()
    opens[-1] = 99.0                                       # -1% down-gap
    sig = asyncio.run(GapFillFadeStrategy().analyze(_df(closes, opens), "SPY"))
    assert sig is not None and sig.side == "buy"
    opens[-1] = 101.0                                      # +1% up-gap
    sig = asyncio.run(GapFillFadeStrategy().analyze(_df(closes, opens), "SPY"))
    assert sig is not None and sig.side == "sell"


def test_gap_fade_skips_noise_news_and_crisis():
    closes = np.full(40, 100.0)
    opens = closes.copy()
    opens[-1] = 100.1                                      # 0.1% — noise
    assert asyncio.run(GapFillFadeStrategy().analyze(_df(closes, opens), "SPY")) is None
    opens[-1] = 105.0                                      # 5% — news, don't fade
    assert asyncio.run(GapFillFadeStrategy().analyze(_df(closes, opens), "SPY")) is None
    rs = np.random.RandomState(1)                          # crisis vol regime
    crisis = 100 * np.exp(np.cumsum(rs.normal(0, 0.05, 40)))
    o = crisis.copy(); o[-1] = crisis[-2] * 0.99
    assert asyncio.run(GapFillFadeStrategy().analyze(_df(crisis, o), "SPY")) is None


# ── double_seven ──────────────────────────────────────────────────────────────

def test_double7_buys_7day_low_in_uptrend_only():
    up = np.linspace(100, 130, 220)
    up[-1] = up[-8:-1].min() - 0.5                          # fresh 7-day low
    sig = asyncio.run(DoubleSevenStrategy().analyze(_df(up), "SPY"))
    assert sig is not None and sig.side == "buy"
    down = np.linspace(130, 100, 220)                       # below 200-SMA
    down[-1] = down[-8:-1].min() - 0.5
    assert asyncio.run(DoubleSevenStrategy().analyze(_df(down), "SPY")) is None


def test_double7_none_when_not_at_low():
    up = np.linspace(100, 130, 220)                         # last bar is the high
    assert asyncio.run(DoubleSevenStrategy().analyze(_df(up), "SPY")) is None


def test_backtest_signals_no_lookahead_shapes():
    df = _df(np.linspace(100, 120, 250))
    for cls in (TurnOfMonthStrategy, GapFillFadeStrategy, DoubleSevenStrategy):
        bs = cls().backtest_signals(df)
        assert len(bs.entries) == len(df) and bs.entries.dtype == bool
        assert len(bs.exits) == len(df)
