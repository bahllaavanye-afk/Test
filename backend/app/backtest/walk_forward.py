"""Walk-forward validation: train on N years, test on M months, roll forward."""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from app.backtest.engine import run_backtest, BacktestMetrics

TIMEFRAME_TRAIN = 2  # years of training data
TIMEFRAME_TEST = 6  # months of testing data

MAX_EQUIITY = 100_000

@dataclass
class WalkForwardResult:
    windows: list[dict] = field(default_factory=list)
    avg_sharpe: float = 0.0
    avg_drawdown: float = 0.0
    combined_equity: list[dict] = field(default_factory=list)


def walk_forward(
    signals_fn,               # callable(train_df, test_df) -> pd.Series of signals on test_df
    prices: pd.Series,
    train_years: int | None = None,
    test_months: int | None = None,
) -> WalkForwardResult:
    """
    Rolls a train/test window across entire history.
    signals_fn receives (train_prices, test_prices) and must return signals for test period only.
    """
    train_bars = (train_years if train_years is not None else TIMEFRAME_TRAIN) * 252
    test_bars = (test_months if test_months is not None else TIMEFRAME_TEST) * 21
    result = WalkForwardResult()
    equity_carry = MAX_EQUIITY

    i = train_bars
    while i + test_bars <= len(prices):
        train = prices.iloc[i - train_bars:i]
        test = prices.iloc[i:i + test_bars]

        try:
            test_signals = signals_fn(train, test)
            metrics = run_backtest(test_signals, test, initial_equity=equity_carry)
            equity_carry = metrics.equity_curve[-1]["equity"] if metrics.equity_curve else equity_carry

            result.windows.append({
                "start": str(test.index[0].date()),
                "end": str(test.index[-1].date()),
                "sharpe": metrics.sharpe,
                "max_drawdown": metrics.max_drawdown,
                "total_return": metrics.total_return,
                "num_trades": metrics.num_trades,
            })
            result.combined_equity.extend(metrics.equity_curve)
        except Exception as e:
            result.windows.append({"start": str(test.index[0].date()), "end": str(test.index[-1].date()), "error": str(e)})

        i += test_bars

    sharpes = [w["sharpe"] for w in result.windows if "sharpe" in w]
    dds = [w["max_drawdown"] for w in result.windows if "max_drawdown" in w]
    result.avg_sharpe = round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0
    result.avg_drawdown = round(sum(dds) / len(dds), 4) if dds else 0.0
    return result


# ==================== Unit Tests ====================

import pytest
from datetime import datetime, timedelta


def _generate_price_series(length: int) -> pd.Series:
    """Helper to create a price series with a daily datetime index."""
    start = datetime(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(length)]
    # simple increasing price to avoid NaNs
    prices = pd.Series(range(1, length + 1), index=pd.DatetimeIndex(dates))
    return prices


def _dummy_metrics(equity_end: float = MAX_EQUIITY) -> BacktestMetrics:
    """Create a minimal BacktestMetrics-like object for testing."""
    class DummyMetrics:
        def __init__(self):
            self.equity_curve = [{"equity": equity_end}]
            self.sharpe = 1.23
            self.max_drawdown = -0.05
            self.total_return = 0.10
            self.num_trades = 5
    return DummyMetrics()


def test_single_window_exact_length(monkeypatch):
    """
    Edge case: price series length is exactly train_bars + test_bars.
    Expect a single window and correct averaging.
    """
    train_years = 1
    test_months = 1
    train_bars = train_years * 252
    test_bars = test_months * 21
    prices = _generate_price_series(train_bars + test_bars)

    def dummy_signals(train, test):
        # return a series of zeros matching test length
        return pd.Series(0, index=test.index)

    monkeypatch.setattr("app.backtest.engine.run_backtest", lambda sig, prc, initial_equity: _dummy_metrics())
    result = walk_forward(dummy_signals, prices, train_years=train_years, test_months=test_months)

    assert len(result.windows) == 1
    assert result.avg_sharpe == 1.23
    assert result.avg_drawdown == -0.05


def test_no_window_when_insufficient_data(monkeypatch):
    """
    Edge case: price series shorter than required train + test bars.
    Expect zero windows and zero averages.
    """
    train_years = 2
    test_months = 6
    train_bars = train_years * 252
    test_bars = test_months * 21
    # one less than needed
    prices = _generate_price_series(train_bars + test_bars - 1)

    def dummy_signals(train, test):
        return pd.Series(0, index=test.index)

    monkeypatch.setattr("app.backtest.engine.run_backtest", lambda sig, prc, initial_equity: _dummy_metrics())
    result = walk_forward(dummy_signals, prices, train_years=train_years, test_months=test_months)

    assert result.windows == []
    assert result.avg_sharpe == 0.0
    assert result.avg_drawdown == 0.0


def test_signal_function_exception_handling(monkeypatch):
    """
    Edge case: signals_fn raises an exception.
    Verify that the error is captured in the window dict and does not break the loop.
    """
    train_years = 1
    test_months = 1
    train_bars = train_years * 252
    test_bars = test_months * 21
    # Provide enough data for two windows
    prices = _generate_price_series(train_bars + 2 * test_bars)

    call_count = {"cnt": 0}

    def faulty_signals(train, test):
        call_count["cnt"] += 1
        if call_count["cnt"] == 1:
            raise ValueError("Intentional failure")
        return pd.Series(0, index=test.index)

    monkeypatch.setattr("app.backtest.engine.run_backtest", lambda sig, prc, initial_equity: _dummy_metrics())
    result = walk_forward(faulty_signals, prices, train_years=train_years, test_months=test_months)

    # First window should contain an error entry, second should be successful
    assert len(result.windows) == 2
    assert "error" in result.windows[0]
    assert result.windows[0]["error"] == "Intentional failure"
    assert "sharpe" in result.windows[1]
    assert result.avg_sharpe == 1.23
    assert result.avg_drawdown == -0.05