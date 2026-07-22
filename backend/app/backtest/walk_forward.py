"""Walk-forward validation: train on N years, test on M months, roll forward."""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any

from app.backtest.engine import run_backtest, BacktestMetrics
from app.backtest.cpcv import deflated_sharpe_ratio

# Constants
TIMEFRAME_TRAIN = 2  # years of training data
TIMEFRAME_TEST = 6  # months of testing data

# Overfit gate thresholds — the protocol documented in this module's CLAUDE.md
# ("OOS Sharpe ≥ 0.7 across 12+ windows"), now ENFORCED as a computed verdict
# instead of a comment. DSR>0 adds the multiple-testing haircut on top.
MIN_WINDOWS = 12       # ≥ 1 year of OOS at 1-month steps
MIN_OOS_SHARPE = 0.7   # per-window and average bar
MIN_CONSISTENCY = 0.5  # ≥ half the windows must clear the per-window bar
MIN_DSR = 0.90         # Deflated Sharpe probability (multiple-testing haircut)

MAX_EQUITY = 100_000

DAYS_PER_YEAR = 252
DAYS_PER_MONTH = 21

KEY_START = "start"
KEY_END = "end"
KEY_SHARPE = "sharpe"
KEY_MAX_DRAWDOWN = "max_drawdown"
KEY_TOTAL_RETURN = "total_return"
KEY_NUM_TRADES = "num_trades"
KEY_ERROR = "error"


@dataclass
class WalkForwardResult:
    windows: List[Dict[str, Any]] = field(default_factory=list)
    avg_sharpe: float = 0.0
    avg_drawdown: float = 0.0
    combined_equity: List[Dict[str, Any]] = field(default_factory=list)
    # Overfit gate (populated by walk_forward()):
    n_windows: int = 0
    deflated_sharpe: float = 0.0   # DSR over the window Sharpes (multiple-testing haircut)
    consistency: float = 0.0       # fraction of windows with Sharpe ≥ MIN_OOS_SHARPE
    is_robust: bool = False        # passes the full protocol → safe to promote
    verdict: str = "insufficient_data"


def robustness_verdict(sharpes: List[float]) -> Dict[str, Any]:
    """Grade a walk-forward's per-window Sharpes against the documented protocol.

    Pure + side-effect free so it can be unit-tested directly. A strategy is
    ``robust`` only if it has enough OOS windows, an average and a majority of
    windows clearing the per-window bar, AND a positive Deflated Sharpe (so a
    handful of lucky windows can't carry it past the multiple-testing haircut).
    """
    n = len(sharpes)
    if n == 0:
        return {
            "n_windows": 0,
            "deflated_sharpe": 0.0,
            "consistency": 0.0,
            "is_robust": False,
            "verdict": "insufficient_data",
        }

    avg = sum(sharpes) / n
    consistency = sum(1 for s in sharpes if s >= MIN_OOS_SHARPE) / n
    dsr = deflated_sharpe_ratio(sharpes, n_trials=n)

    reasons: List[str] = []
    if n < MIN_WINDOWS:
        reasons.append(f"only {n} windows (<{MIN_WINDOWS})")
    if avg < MIN_OOS_SHARPE:
        reasons.append(f"avg Sharpe {avg:.2f} (<{MIN_OOS_SHARPE})")
    if consistency < MIN_CONSISTENCY:
        reasons.append(f"consistency {consistency:.0%} (<{MIN_CONSISTENCY:.0%})")
    if dsr < MIN_DSR:
        reasons.append(f"DSR {dsr:.2f} (<{MIN_DSR} — within luck)")

    is_robust = not reasons
    verdict = "robust" if is_robust else "overfit_or_weak: " + "; ".join(reasons)

    return {
        "n_windows": n,
        "deflated_sharpe": round(dsr, 4),
        "consistency": round(consistency, 4),
        "is_robust": is_robust,
        "verdict": verdict,
    }


def walk_forward(
    signals_fn: Callable[[pd.Series, pd.Series], pd.Series],
    prices: pd.Series,
    train_years: int | None = None,
    test_months: int | None = None,
    initial_equity: float | None = None,
) -> WalkForwardResult:
    """
    Rolls a train/test window across entire history.
    signals_fn receives (train_prices, test_prices) and must return signals for test period only.
    """
    train_bars = (train_years if train_years is not None else TIMEFRAME_TRAIN) * DAYS_PER_YEAR
    test_bars = (test_months if test_months is not None else TIMEFRAME_TEST) * DAYS_PER_MONTH

    result = WalkForwardResult()
    equity_carry = initial_equity if initial_equity is not None else MAX_EQUITY

    # Running sums for incremental averaging
    sharpe_sum = 0.0
    drawdown_sum = 0.0
    sharpe_count = 0
    drawdown_count = 0
    sharpes: List[float] = []

    i = train_bars
    max_index = len(prices)
    while i + test_bars <= max_index:
        train = prices.iloc[i - train_bars : i]
        test = prices.iloc[i : i + test_bars]

        window_info: Dict[str, Any] = {
            KEY_START: str(test.index[0].date()),
            KEY_END: str(test.index[-1].date()),
        }

        try:
            test_signals = signals_fn(train, test)
            metrics: BacktestMetrics = run_backtest(test_signals, test, initial_equity=equity_carry)

            # Update equity carry for next window
            equity_carry = metrics.equity_curve[-1]["equity"] if metrics.equity_curve else equity_carry

            # Record metrics
            window_info.update(
                {
                    KEY_SHARPE: metrics.sharpe,
                    KEY_MAX_DRAWDOWN: metrics.max_drawdown,
                    KEY_TOTAL_RETURN: metrics.total_return,
                    KEY_NUM_TRADES: metrics.num_trades,
                }
            )
            result.combined_equity.extend(metrics.equity_curve)

            # Incremental aggregation
            sharpe_sum += metrics.sharpe
            drawdown_sum += metrics.max_drawdown
            sharpe_count += 1
            drawdown_count += 1
            sharpes.append(metrics.sharpe)

        except Exception as e:
            window_info[KEY_ERROR] = str(e)

        result.windows.append(window_info)
        i += test_bars

    # Final averages using the incremental sums
    result.avg_sharpe = round(sharpe_sum / sharpe_count, 4) if sharpe_count else 0.0
    result.avg_drawdown = round(drawdown_sum / drawdown_count, 4) if drawdown_count else 0.0

    # Overfit gate: grade the OOS window distribution (DSR + consistency).
    verdict = robustness_verdict(sharpes)
    result.n_windows = verdict["n_windows"]
    result.deflated_sharpe = verdict["deflated_sharpe"]
    result.consistency = verdict["consistency"]
    result.is_robust = verdict["is_robust"]
    result.verdict = verdict["verdict"]
    return result