"""Unit tests for backtest engine."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from app.backtest.engine import run_backtest


def make_prices(n=500, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    prices = 100 * np.cumprod(1 + returns)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx)


def test_backtest_buy_and_hold():
    prices = make_prices()
    signals = pd.Series(1, index=prices.index)
    metrics = run_backtest(signals, prices)
    assert metrics.sharpe is not None
    assert -1.0 <= metrics.max_drawdown <= 0.0
    assert 0 <= metrics.win_rate <= 1.0
    assert len(metrics.equity_curve) > 0


def test_backtest_empty_signals():
    prices = make_prices()
    signals = pd.Series(0, index=prices.index)
    metrics = run_backtest(signals, prices)
    assert metrics.num_trades == 0
