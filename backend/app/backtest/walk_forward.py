"""Walk-forward validation: train on N years, test on M months, roll forward."""

from __future__ import annotations

import collections.abc
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
    # ---- Input validation ----
    if not isinstance(signals_fn, collections.abc.Callable):
        raise ValueError("signals_fn must be a callable that accepts (train_df, test_df) and returns a pd.Series.")
    if not isinstance(prices, pd.Series):
        raise ValueError("prices must be a pandas Series.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices index must be a pandas DatetimeIndex.")
    if prices.empty:
        raise ValueError("prices series cannot be empty.")
    if train_years is not None:
        if not isinstance(train_years, int) or train_years <= 0:
            raise ValueError("train_years must be a positive integer if provided.")
    if test_months is not None:
        if not isinstance(test_months, int) or test_months <= 0:
            raise ValueError("test_months must be a positive integer if provided.")
    # --------------------------

    train_bars = (train_years if train_years is not None else TIMEFRAME_TRAIN) * 252
    test_bars = (test_months if test_months is not None else TIMEFRAME_TEST) * 21

    if train_bars <= 0:
        raise ValueError("Calculated train_bars must be greater than zero.")
    if test_bars <= 0:
        raise ValueError("Calculated test_bars must be greater than zero.")
    if train_bars + test_bars > len(prices):
        raise ValueError(
            "The combined train and test window exceeds the length of the provided price series."
        )

    result = WalkForwardResult()
    equity_carry = MAX_EQUIITY

    i = train_bars
    while i + test_bars <= len(prices):
        train = prices.iloc[i - train_bars:i]
        test = prices.iloc[i:i + test_bars]

        try:
            test_signals = signals_fn(train, test)
            if not isinstance(test_signals, pd.Series):
                raise ValueError("signals_fn must return a pandas Series.")
            if len(test_signals) != len(test):
                raise ValueError(
                    "signals_fn must return a Series with the same length as the test period."
                )
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
            result.windows.append({
                "start": str(test.index[0].date()),
                "end": str(test.index[-1].date()),
                "error": str(e),
            })

        i += test_bars

    sharpes = [w["sharpe"] for w in result.windows if "sharpe" in w]
    dds = [w["max_drawdown"] for w in result.windows if "max_drawdown" in w]
    result.avg_sharpe = round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0
    result.avg_drawdown = round(sum(dds) / len(dds), 4) if dds else 0.0
    return result