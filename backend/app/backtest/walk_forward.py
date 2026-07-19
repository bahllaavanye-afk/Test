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

    # Early exit if data insufficient
    if len(prices) < train_bars + test_bars:
        return WalkForwardResult()

    result = WalkForwardResult()
    equity_carry = MAX_EQUIITY

    # Use a for‑loop for clearer iteration and marginal speed gain
    for start in range(train_bars, len(prices) - test_bars + 1, test_bars):
        train = prices.iloc[start - train_bars : start]
        test = prices.iloc[start : start + test_bars]

        try:
            test_signals = signals_fn(train, test)
            metrics: BacktestMetrics = run_backtest(
                test_signals, test, initial_equity=equity_carry
            )
            # Preserve equity for next window
            if metrics.equity_curve:
                equity_carry = metrics.equity_curve[-1]["equity"]

            window_info = {
                "start": test.index[0].date().isoformat(),
                "end": test.index[-1].date().isoformat(),
                "sharpe": metrics.sharpe,
                "max_drawdown": metrics.max_drawdown,
                "total_return": metrics.total_return,
                "num_trades": metrics.num_trades,
            }
            result.windows.append(window_info)
            result.combined_equity.extend(metrics.equity_curve)
        except Exception as e:
            result.windows.append(
                {
                    "start": test.index[0].date().isoformat(),
                    "end": test.index[-1].date().isoformat(),
                    "error": str(e),
                }
            )

    # Compute averages only once after loop
    sharpe_vals = [w["sharpe"] for w in result.windows if "sharpe" in w]
    drawdown_vals = [w["max_drawdown"] for w in result.windows if "max_drawdown" in w]

    result.avg_sharpe = round(sum(sharpe_vals) / len(sharpe_vals), 4) if sharpe_vals else 0.0
    result.avg_drawdown = (
        round(sum(drawdown_vals) / len(drawdown_vals), 4) if drawdown_vals else 0.0
    )
    return result