"""Walk-forward validation: train on N years, test on M months, roll forward."""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field

from app.backtest.engine import BacktestMetrics, run_backtest

# Default configuration constants
TRAIN_YEARS_DEFAULT: int = 2  # years of training data
TEST_MONTHS_DEFAULT: int = 6  # months of testing data
MAX_EQUITY: int = 100_000

# Calendar constants
DAYS_PER_YEAR: int = 252
DAYS_PER_MONTH: int = 21

# Result dictionary keys
KEY_START: str = "start"
KEY_END: str = "end"
KEY_SHARPE: str = "sharpe"
KEY_MAX_DD: str = "max_drawdown"
KEY_TOTAL_RETURN: str = "total_return"
KEY_NUM_TRADES: str = "num_trades"
KEY_ERROR: str = "error"


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
    train_bars = (train_years if train_years is not None else TRAIN_YEARS_DEFAULT) * DAYS_PER_YEAR
    test_bars = (test_months if test_months is not None else TEST_MONTHS_DEFAULT) * DAYS_PER_MONTH

    result = WalkForwardResult()
    equity_carry = MAX_EQUITY

    i = train_bars
    while i + test_bars <= len(prices):
        train = prices.iloc[i - train_bars : i]
        test = prices.iloc[i : i + test_bars]

        try:
            test_signals = signals_fn(train, test)
            metrics = run_backtest(test_signals, test, initial_equity=equity_carry)
            equity_carry = metrics.equity_curve[-1]["equity"] if metrics.equity_curve else equity_carry

            result.windows.append(
                {
                    KEY_START: str(test.index[0].date()),
                    KEY_END: str(test.index[-1].date()),
                    KEY_SHARPE: metrics.sharpe,
                    KEY_MAX_DD: metrics.max_drawdown,
                    KEY_TOTAL_RETURN: metrics.total_return,
                    KEY_NUM_TRADES: metrics.num_trades,
                }
            )
            result.combined_equity.extend(metrics.equity_curve)
        except Exception as e:
            result.windows.append(
                {
                    KEY_START: str(test.index[0].date()),
                    KEY_END: str(test.index[-1].date()),
                    KEY_ERROR: str(e),
                }
            )

        i += test_bars

    sharpes = [w[KEY_SHARPE] for w in result.windows if KEY_SHARPE in w]
    dds = [w[KEY_MAX_DD] for w in result.windows if KEY_MAX_DD in w]
    result.avg_sharpe = round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0
    result.avg_drawdown = round(sum(dds) / len(dds), 4) if dds else 0.0
    return result