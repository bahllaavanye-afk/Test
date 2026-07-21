"""Regression guard: the vectorized backtest engine must never earn a return that
requires knowing the future (look-ahead bias). This is the property the whole
"walk-forward only" mandate rests on — a leak here silently promotes overfit
strategies. These tests lock the property in so a future refactor can't reintroduce it.

Audit note (2026-07-20): the engine is look-ahead-FREE — `position = signal.shift(1)`,
and P&L = position × forward bar-return, so a position is only ever earned on the bar
AFTER its signal. (Separately, callers' strategies also `.shift(1)` their features for
look-ahead avoidance, giving a conservative 2-bar execution lag that *understates*
returns — the safe direction; changing that needs A/B backtest validation, not a
casual edit.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas")

from app.backtest.engine import run_backtest  # noqa: E402


def _prices_with_spike(n: int = 60, spike_at: int = 30, jump: float = 0.20) -> pd.Series:
    """Flat prices with a single large up-move between spike_at-1 and spike_at."""
    p = np.full(n, 100.0)
    p[spike_at:] = 100.0 * (1 + jump)   # everything from spike_at on is +20%
    return pd.Series(p, index=pd.RangeIndex(n))


def test_signal_on_the_spike_bar_cannot_capture_it():
    # A signal placed EXACTLY on the jump bar could only profit if the engine
    # used the same-bar (future) return. A look-ahead-free engine earns ~0.
    prices = _prices_with_spike()
    signals = pd.Series(0.0, index=prices.index)
    signals.iloc[30] = 1.0            # long only on the spike bar itself
    m = run_backtest(signals, prices, fill_at_open=False, commission_pct=0, slippage_pct=0)
    # The +20% move happened INTO bar 30; a clean engine holds nothing during it.
    assert m.total_return < 0.02, f"look-ahead leak: captured {m.total_return:.2%} of a same-bar spike"


def test_signal_before_the_spike_does_capture_it():
    # Positive control: a signal placed a couple of bars BEFORE the jump is
    # legitimately positioned and MUST earn the move (engine isn't just dead).
    prices = _prices_with_spike()
    signals = pd.Series(0.0, index=prices.index)
    signals.iloc[25:35] = 1.0          # long across the jump
    m = run_backtest(signals, prices, fill_at_open=False, commission_pct=0, slippage_pct=0)
    assert m.total_return > 0.15, f"legit move not captured: {m.total_return:.2%}"


def test_all_flat_prices_zero_return():
    prices = pd.Series(np.full(40, 100.0), index=pd.RangeIndex(40))
    signals = pd.Series(1.0, index=prices.index)   # always long, but nothing moves
    m = run_backtest(signals, prices, commission_pct=0, slippage_pct=0)
    assert abs(m.total_return) < 1e-9


def test_position_earns_only_the_bar_after_signal():
    # Direct property check: one-bar signal at t → the only P&L bar is t+1.
    prices = pd.Series([100, 100, 110, 110, 110], index=pd.RangeIndex(5), dtype=float)
    signals = pd.Series([0, 1, 0, 0, 0], index=pd.RangeIndex(5), dtype=float)
    # Signal at bar1 → position at bar2 → earns the 100→110 move landing at bar2.
    m = run_backtest(signals, prices, fill_at_open=False, commission_pct=0, slippage_pct=0)
    assert m.total_return > 0.09  # ~10% move captured, one bar after the signal
