"""Unit tests for backtest engine with enhanced strategy signal validation."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.backtest.engine import run_backtest


def make_prices(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate a synthetic price series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    prices = 100 * np.cumprod(1 + returns)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate a simple RSI."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()

    rs = roll_up / roll_down.replace(to_replace=0, method="bfill")
    return 100 - (100 / (1 + rs))


def generate_signals(prices: pd.Series) -> pd.Series:
    """
    Generate long entry signals using tightened entry logic:
    - Price must be above its 20‑day SMA (trend filter)
    - RSI must be above 55 (momentum filter)
    - Signal is 1 only when both conditions hold, otherwise 0.
    """
    sma20 = prices.rolling(window=20, min_periods=20).mean()
    rsi14 = rsi(prices, window=14)

    condition = (prices > sma20) & (rsi14 > 55)
    return pd.Series(np.where(condition, 1, 0), index=prices.index)


def test_backtest_buy_and_hold():
    """Baseline test: always‑long position."""
    prices = make_prices()
    signals = pd.Series(1, index=prices.index)
    metrics = run_backtest(signals, prices)

    assert metrics.sharpe is not None
    assert -1.0 <= metrics.max_drawdown <= 0.0
    assert 0.0 <= metrics.win_rate <= 1.0
    assert len(metrics.equity_curve) > 0


def test_backtest_empty_signals():
    """Zero‑signal scenario should produce no trades."""
    prices = make_prices()
    signals = pd.Series(0, index=prices.index)
    metrics = run_backtest(signals, prices)

    assert metrics.num_trades == 0


def test_backtest_tightened_entry_conditions():
    """
    Verify that the tightened entry logic yields fewer trades than a naive
    always‑long signal while still producing a valid equity curve.
    """
    prices = make_prices()
    naive_signals = pd.Series(1, index=prices.index)
    refined_signals = generate_signals(prices)

    naive_metrics = run_backtest(naive_signals, prices)
    refined_metrics = run_backtest(refined_signals, prices)

    # The refined strategy should trade less frequently.
    assert refined_metrics.num_trades < naive_metrics.num_trades

    # Ensure the refined strategy still generates a non‑empty equity curve.
    assert len(refined_metrics.equity_curve) > 0

    # Basic sanity checks on the refined metrics.
    assert refined_metrics.sharpe is not None
    assert -1.0 <= refined_metrics.max_drawdown <= 0.0
    assert 0.0 <= refined_metrics.win_rate <= 1.0


def test_backtest_no_trade_when_conditions_never_met():
    """
    Create a flat price series where the SMA and RSI conditions cannot be satisfied.
    The backtest should report zero trades.
    """
    flat_prices = pd.Series(100.0, index=pd.date_range("2020-01-01", periods=100, freq="D"))
    signals = generate_signals(flat_prices)
    metrics = run_backtest(signals, flat_prices)

    assert metrics.num_trades == 0
    # Equity curve should consist only of the initial capital.
    assert len(metrics.equity_curve) == len(flat_prices)  # equity curve always aligns with price index.