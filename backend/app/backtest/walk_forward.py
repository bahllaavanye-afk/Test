"""Walk-forward validation: train on N years, test on M months, roll forward."""

from __future__ import annotations

import logging
import time
import pandas as pd
from dataclasses import dataclass, field
from app.backtest.engine import run_backtest, BacktestMetrics

# Configure module logger
logger = logging.getLogger(__name__)

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

        start_date = str(test.index[0].date())
        end_date = str(test.index[-1].date())

        try:
            # Generate signals
            test_signals = signals_fn(train, test)

            # Run backtest and measure execution time
            start_time = time.perf_counter()
            metrics = run_backtest(test_signals, test, initial_equity=equity_carry)
            exec_time = time.perf_counter() - start_time

            # Update equity carry forward
            equity_carry = (
                metrics.equity_curve[-1]["equity"]
                if metrics.equity_curve
                else equity_carry
            )

            # Log key metrics
            signal_count = test_signals.shape[0] if isinstance(test_signals, pd.Series) else len(test_signals)
            logger.info(
                "Walk-forward window %s - %s | signals=%d | exec_time=%.4fs | pnl=%.4f",
                start_date,
                end_date,
                signal_count,
                exec_time,
                metrics.total_return,
            )

            # Record window results
            result.windows.append({
                "start": start_date,
                "end": end_date,
                "sharpe": metrics.sharpe,
                "max_drawdown": metrics.max_drawdown,
                "total_return": metrics.total_return,
                "num_trades": metrics.num_trades,
            })
            result.combined_equity.extend(metrics.equity_curve)

        except Exception as e:
            logger.error(
                "Walk-forward window %s - %s encountered error: %s",
                start_date,
                end_date,
                e,
                exc_info=True,
            )
            result.windows.append({
                "start": start_date,
                "end": end_date,
                "error": str(e),
            })

        i += test_bars

    # Compute aggregate statistics and log them
    sharpes = [w["sharpe"] for w in result.windows if "sharpe" in w]
    dds = [w["max_drawdown"] for w in result.windows if "max_drawdown" in w]
    result.avg_sharpe = round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0
    result.avg_drawdown = round(sum(dds) / len(dds), 4) if dds else 0.0

    logger.info(
        "Walk-forward completed | avg_sharpe=%.4f | avg_drawdown=%.4f",
        result.avg_sharpe,
        result.avg_drawdown,
    )

    return result