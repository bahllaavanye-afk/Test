"""Walk-forward validation: train on N years, test on M months, roll forward."""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Tuple

from app.backtest.engine import run_backtest, BacktestMetrics

TIMEFRAME_TRAIN = 2  # years of training data
TIMEFRAME_TEST = 6  # months of testing data

MAX_EQUIITY = 100_000


@dataclass
class WalkForwardResult:
    windows: List[Dict[str, Any]] = field(default_factory=list)
    avg_sharpe: float = 0.0
    avg_drawdown: float = 0.0
    combined_equity: List[Dict[str, Any]] = field(default_factory=list)


SignalFn = Callable[[pd.Series, pd.Series], pd.Series]


def _bars_per_year() -> int:
    """Number of trading days assumed per year."""
    return 252


def _bars_per_month() -> int:
    """Number of trading days assumed per month."""
    return 21


def _calculate_window_sizes(
    train_years: int | None,
    test_months: int | None,
) -> Tuple[int, int]:
    """Return the number of bars for training and testing windows."""
    train_bars = (train_years if train_years is not None else TIMEFRAME_TRAIN) * _bars_per_year()
    test_bars = (test_months if test_months is not None else TIMEFRAME_TEST) * _bars_per_month()
    return train_bars, test_bars


def _slice_window(
    prices: pd.Series,
    start_idx: int,
    train_bars: int,
    test_bars: int,
) -> Tuple[pd.Series, pd.Series]:
    """Slice the price series into training and testing segments."""
    train = prices.iloc[start_idx - train_bars : start_idx]
    test = prices.iloc[start_idx : start_idx + test_bars]
    return train, test


def _run_backtest_window(
    signals_fn: SignalFn,
    train: pd.Series,
    test: pd.Series,
    equity_carry: float,
) -> Tuple[BacktestMetrics, float]:
    """
    Execute backtest for a single window and return the metrics together with the updated equity.
    """
    test_signals = signals_fn(train, test)
    metrics = run_backtest(test_signals, test, initial_equity=equity_carry)
    new_equity = (
        metrics.equity_curve[-1]["equity"]
        if metrics.equity_curve
        else equity_carry
    )
    return metrics, new_equity


def _record_success(
    result: WalkForwardResult,
    test: pd.Series,
    metrics: BacktestMetrics,
    equity_carry: float,
) -> None:
    """Append successful window metrics to the result."""
    result.windows.append(
        {
            "start": str(test.index[0].date()),
            "end": str(test.index[-1].date()),
            "sharpe": metrics.sharpe,
            "max_drawdown": metrics.max_drawdown,
            "total_return": metrics.total_return,
            "num_trades": metrics.num_trades,
        }
    )
    result.combined_equity.extend(metrics.equity_curve)


def _record_error(
    result: WalkForwardResult,
    test: pd.Series,
    error: Exception,
) -> None:
    """Append error information for a window to the result."""
    result.windows.append(
        {
            "start": str(test.index[0].date()),
            "end": str(test.index[-1].date()),
            "error": str(error),
        }
    )


def _aggregate_averages(result: WalkForwardResult) -> None:
    """Compute average Sharpe and drawdown across all successful windows."""
    sharpe_values = [w["sharpe"] for w in result.windows if "sharpe" in w]
    drawdown_values = [w["max_drawdown"] for w in result.windows if "max_drawdown" in w]

    result.avg_sharpe = round(sum(sharpe_values) / len(sharpe_values), 4) if sharpe_values else 0.0
    result.avg_drawdown = round(sum(drawdown_values) / len(drawdown_values), 4) if drawdown_values else 0.0


def walk_forward(
    signals_fn: SignalFn,               # callable(train_df, test_df) -> pd.Series of signals on test_df
    prices: pd.Series,
    train_years: int | None = None,
    test_months: int | None = None,
) -> WalkForwardResult:
    """
    Rolls a train/test window across the entire price history.

    Parameters
    ----------
    signals_fn : callable
        Function that receives (train_prices, test_prices) and returns a Series of signals for the test period.
    prices : pd.Series
        Historical price series indexed by datetime.
    train_years : int, optional
        Number of years to use for the training window. Falls back to TIMEFRAME_TRAIN if None.
    test_months : int, optional
        Number of months to use for the testing window. Falls back to TIMEFRAME_TEST if None.

    Returns
    -------
    WalkForwardResult
        Aggregated metrics and equity curve across all windows.
    """
    train_bars, test_bars = _calculate_window_sizes(train_years, test_months)
    result = WalkForwardResult()
    equity_carry = MAX_EQUIITY

    i = train_bars
    while i + test_bars <= len(prices):
        train, test = _slice_window(prices, i, train_bars, test_bars)

        try:
            metrics, equity_carry = _run_backtest_window(signals_fn, train, test, equity_carry)
            _record_success(result, test, metrics, equity_carry)
        except Exception as exc:  # pragma: no cover
            _record_error(result, test, exc)

        i += test_bars

    _aggregate_averages(result)
    return result