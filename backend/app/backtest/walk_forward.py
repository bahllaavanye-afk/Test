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


def _slice_window(
    prices: pd.Series,
    start_idx: int,
    train_bars: int,
    test_bars: int,
) -> tuple[pd.Series, pd.Series]:
    """Return training and testing slices for the given start index."""
    train = prices.iloc[start_idx - train_bars : start_idx]
    test = prices.iloc[start_idx : start_idx + test_bars]
    return train, test


def _process_window(
    signals_fn,
    train: pd.Series,
    test: pd.Series,
    equity_carry: float,
) -> tuple[BacktestMetrics | None, float, dict]:
    """
    Run backtest for a single window.

    Returns:
        metrics: BacktestMetrics if successful, otherwise None.
        new_equity: Updated equity after this window.
        window_info: Dictionary with window metadata or error information.
    """
    try:
        test_signals = signals_fn(train, test)
        metrics = run_backtest(test_signals, test, initial_equity=equity_carry)
        new_equity = (
            metrics.equity_curve[-1]["equity"]
            if metrics.equity_curve
            else equity_carry
        )
        window_info = {
            "start": str(test.index[0].date()),
            "end": str(test.index[-1].date()),
            "sharpe": metrics.sharpe,
            "max_drawdown": metrics.max_drawdown,
            "total_return": metrics.total_return,
            "num_trades": metrics.num_trades,
        }
        return metrics, new_equity, window_info
    except Exception as e:  # pragma: no cover
        window_info = {
            "start": str(test.index[0].date()),
            "end": str(test.index[-1].date()),
            "error": str(e),
        }
        return None, equity_carry, window_info


def _compute_averages(windows: list[dict]) -> tuple[float, float]:
    """Calculate average Sharpe and average drawdown from successful windows."""
    sharpes = [w["sharpe"] for w in windows if "sharpe" in w]
    drawdowns = [w["max_drawdown"] for w in windows if "max_drawdown" in w]

    avg_sharpe = round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0
    avg_drawdown = round(sum(drawdowns) / len(drawdowns), 4) if drawdowns else 0.0
    return avg_sharpe, avg_drawdown


def walk_forward(
    signals_fn,  # callable(train_df, test_df) -> pd.Series of signals on test_df
    prices: pd.Series,
    train_years: int | None = None,
    test_months: int | None = None,
) -> WalkForwardResult:
    """
    Rolls a train/test window across entire history.

    signals_fn receives (train_prices, test_prices) and must return signals for the test period only.
    """
    train_bars = (train_years if train_years is not None else TIMEFRAME_TRAIN) * 252
    test_bars = (test_months if test_months is not None else TIMEFRAME_TEST) * 21

    result = WalkForwardResult()
    equity_carry = MAX_EQUIITY

    start_idx = train_bars
    while start_idx + test_bars <= len(prices):
        train_slice, test_slice = _slice_window(prices, start_idx, train_bars, test_bars)

        metrics, equity_carry, window_info = _process_window(
            signals_fn, train_slice, test_slice, equity_carry
        )

        result.windows.append(window_info)
        if metrics:
            result.combined_equity.extend(metrics.equity_curve)

        start_idx += test_bars

    result.avg_sharpe, result.avg_drawdown = _compute_averages(result.windows)
    return result