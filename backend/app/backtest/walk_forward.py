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


def _get_window_lengths(
    train_years: int | None,
    test_months: int | None,
) -> tuple[int, int]:
    """Return the number of bars for training and testing windows."""
    train_bars = (train_years if train_years is not None else TIMEFRAME_TRAIN) * 252
    test_bars = (test_months if test_months is not None else TIMEFRAME_TEST) * 21
    return train_bars, test_bars


def _slice_data(
    prices: pd.Series,
    start_idx: int,
    train_bars: int,
    test_bars: int,
) -> tuple[pd.Series, pd.Series]:
    """Slice the price series into training and testing segments."""
    train = prices.iloc[start_idx - train_bars : start_idx]
    test = prices.iloc[start_idx : start_idx + test_bars]
    return train, test


def _run_window(
    signals_fn,
    train: pd.Series,
    test: pd.Series,
    equity_carry: float,
) -> tuple[BacktestMetrics | None, float, Exception | None]:
    """
    Execute a single walk‑forward window.

    Returns:
        metrics – BacktestMetrics if successful, otherwise None
        new_equity – updated equity value (unchanged on error)
        error – exception instance if one occurred, otherwise None
    """
    try:
        test_signals = signals_fn(train, test)
        metrics = run_backtest(test_signals, test, initial_equity=equity_carry)
        new_equity = (
            metrics.equity_curve[-1]["equity"] if metrics.equity_curve else equity_carry
        )
        return metrics, new_equity, None
    except Exception as exc:  # pragma: no cover
        return None, equity_carry, exc


def _record_window(
    result: WalkForwardResult,
    test: pd.Series,
    metrics: BacktestMetrics | None,
    error: Exception | None,
) -> float:
    """Append window information to the result and return the equity to carry forward."""
    window_info: dict = {
        "start": str(test.index[0].date()),
        "end": str(test.index[-1].date()),
    }

    if error is None and metrics is not None:
        window_info.update(
            {
                "sharpe": metrics.sharpe,
                "max_drawdown": metrics.max_drawdown,
                "total_return": metrics.total_return,
                "num_trades": metrics.num_trades,
            }
        )
        result.combined_equity.extend(metrics.equity_curve)
        equity = (
            metrics.equity_curve[-1]["equity"] if metrics.equity_curve else MAX_EQUIITY
        )
    else:
        window_info["error"] = str(error) if error else "unknown error"
        equity = MAX_EQUIITY

    result.windows.append(window_info)
    return equity


def _compute_averages(windows: list[dict]) -> tuple[float, float]:
    """Calculate average Sharpe and drawdown from successful windows."""
    sharpe_vals = [w["sharpe"] for w in windows if "sharpe" in w]
    drawdown_vals = [w["max_drawdown"] for w in windows if "max_drawdown" in w]

    avg_sharpe = round(sum(sharpe_vals) / len(sharpe_vals), 4) if sharpe_vals else 0.0
    avg_drawdown = round(sum(drawdown_vals) / len(drawdown_vals), 4) if drawdown_vals else 0.0
    return avg_sharpe, avg_drawdown


def walk_forward(
    signals_fn,  # callable(train_df, test_df) -> pd.Series of signals on test_df
    prices: pd.Series,
    train_years: int | None = None,
    test_months: int | None = None,
) -> WalkForwardResult:
    """
    Rolls a train/test window across entire history.

    `signals_fn` receives (train_prices, test_prices) and must return signals for the test period only.
    """
    train_bars, test_bars = _get_window_lengths(train_years, test_months)
    result = WalkForwardResult()
    equity_carry = MAX_EQUIITY

    i = train_bars
    while i + test_bars <= len(prices):
        train_slice, test_slice = _slice_data(prices, i, train_bars, test_bars)
        metrics, equity_carry, error = _run_window(signals_fn, train_slice, test_slice, equity_carry)
        equity_carry = _record_window(result, test_slice, metrics, error)
        i += test_bars

    result.avg_sharpe, result.avg_drawdown = _compute_averages(result.windows)
    return result