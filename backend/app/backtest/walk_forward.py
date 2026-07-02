"""Walk-forward validation: train on N years, test on M months, roll forward."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List

import pandas as pd

from app.backtest.engine import BacktestMetrics, run_backtest

# Configure module logger
logger = logging.getLogger(__name__)

TIMEFRAME_TRAIN = 2  # years of training data
TIMEFRAME_TEST = 6  # months of testing data

MAX_EQUIITY = 100_000


@dataclass
class WalkForwardResult:
    windows: List[dict] = field(default_factory=list)
    avg_sharpe: float = 0.0
    avg_drawdown: float = 0.0
    combined_equity: List[dict] = field(default_factory=list)


def walk_forward(
    signals_fn: Callable[[pd.Series, pd.Series], pd.Series],
    prices: pd.Series,
    train_years: int | None = None,
    test_months: int | None = None,
) -> WalkForwardResult:
    """
    Rolls a train/test window across the entire price history.

    Parameters
    ----------
    signals_fn : Callable
        Function that receives ``train`` and ``test`` price series and returns a
        ``pd.Series`` of signals for the test period.
    prices : pd.Series
        Time‑indexed price series (must be monotonic increasing index).
    train_years : int | None
        Number of years to use for the training window. Falls back to
        ``TIMEFRAME_TRAIN`` when ``None``.
    test_months : int | None
        Number of months to use for the testing window. Falls back to
        ``TIMEFRAME_TEST`` when ``None``.

    Returns
    -------
    WalkForwardResult
        Aggregated results across all windows, including average Sharpe,
        average drawdown and a combined equity curve.
    """
    # Basic input validation
    if not callable(signals_fn):
        raise TypeError("signals_fn must be a callable.")
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series.")
    if prices.empty:
        raise ValueError("prices series is empty.")
    if not hasattr(prices.index, "date"):
        raise ValueError("prices index must be datetime-like with a .date() method.")

    train_bars = (train_years if train_years is not None else TIMEFRAME_TRAIN) * 252
    test_bars = (test_months if test_months is not None else TIMEFRAME_TEST) * 21

    result = WalkForwardResult()
    equity_carry = MAX_EQUIITY

    i = train_bars
    while i + test_bars <= len(prices):
        train = prices.iloc[i - train_bars : i]
        test = prices.iloc[i : i + test_bars]

        try:
            test_signals = signals_fn(train, test)

            if not isinstance(test_signals, pd.Series):
                raise TypeError(
                    "signals_fn must return a pandas Series, got "
                    f"{type(test_signals).__name__}"
                )
            if len(test_signals) != len(test):
                raise ValueError(
                    f"Signal length ({len(test_signals)}) does not match test period length ({len(test)})."
                )

            metrics: BacktestMetrics = run_backtest(
                test_signals, test, initial_equity=equity_carry
            )
            equity_carry = (
                metrics.equity_curve[-1]["equity"]
                if metrics.equity_curve
                else equity_carry
            )

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

        except (ValueError, TypeError) as e:
            logger.error(
                "Walk-forward window error: %s",
                e,
                extra={"window_start": str(test.index[0]), "window_end": str(test.index[-1])},
            )
            result.windows.append(
                {
                    "start": str(test.index[0].date()),
                    "end": str(test.index[-1].date()),
                    "error": f"{type(e).__name__}: {e}",
                }
            )
        except Exception as e:
            logger.exception(
                "Unexpected error during walk-forward execution.",
                extra={"window_start": str(test.index[0]), "window_end": str(test.index[-1])},
            )
            result.windows.append(
                {
                    "start": str(test.index[0].date()),
                    "end": str(test.index[-1].date()),
                    "error": f"UnexpectedError: {e}",
                }
            )

        i += test_bars

    sharpes = [w["sharpe"] for w in result.windows if "sharpe" in w]
    dds = [w["max_drawdown"] for w in result.windows if "max_drawdown" in w]

    result.avg_sharpe = round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0
    result.avg_drawdown = round(sum(dds) / len(dds), 4) if dds else 0.0

    return result