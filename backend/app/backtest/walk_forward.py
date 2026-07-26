"""Walk‑forward validation utilities.

This module provides functions to evaluate a trading strategy using a
walk‑forward (rolling‑window) scheme.  The main entry point is :func:`walk_forward`,
which iterates over a price series, repeatedly training on a fixed‑length
historical window and testing on a subsequent out‑of‑sample window.  Results
include per‑window performance metrics, aggregated averages, and a robustness
verdict based on the documented over‑fit gate protocol.
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional, Tuple

from app.backtest.engine import run_backtest, BacktestMetrics
from app.backtest.cpcv import deflated_sharpe_ratio

# Constants
TIMEFRAME_TRAIN = 2  # years of training data
TIMEFRAME_TEST = 6  # months of testing data

# Overfit gate thresholds — the protocol documented in this module's CLAUDE.md
# ("OOS Sharpe ≥ 0.7 across 12+ windows"), now ENFORCED as a computed verdict
# instead of a comment. DSR>0 adds the multiple‑testing haircut on top.
MIN_WINDOWS = 12       # ≥ 1 year of OOS at 1‑month steps
MIN_OOS_SHARPE = 0.7   # per‑window and average bar
MIN_CONSISTENCY = 0.5  # ≥ half the windows must clear the per‑window bar
MIN_DSR = 0.90         # Deflated Sharpe probability (multiple‑testing haircut)

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
    """Container for the outcome of a walk‑forward evaluation.

    Attributes
    ----------
    windows: List[Dict[str, Any]]
        Per‑window dictionaries containing performance metrics or error information.
    avg_sharpe: float
        Average Sharpe ratio across windows that reported a Sharpe.
    avg_drawdown: float
        Average maximum drawdown across windows that reported a drawdown.
    combined_equity: List[Dict[str, Any]]
        Concatenated equity‑curve entries from all windows.
    n_windows: int
        Number of out‑of‑sample windows evaluated.
    deflated_sharpe: float
        Deflated Sharpe ratio (DSR) across windows, applying a multiple‑testing haircut.
    consistency: float
        Fraction of windows with Sharpe ≥ ``MIN_OOS_SHARPE``.
    is_robust: bool
        Whether the strategy passes the full robustness protocol.
    verdict: str
        Human‑readable assessment of robustness or reason for failure.
    """
    windows: List[Dict[str, Any]] = field(default_factory=list)
    avg_sharpe: float = 0.0
    avg_drawdown: float = 0.0
    combined_equity: List[Dict[str, Any]] = field(default_factory=list)
    n_windows: int = 0
    deflated_sharpe: float = 0.0
    consistency: float = 0.0
    is_robust: bool = False
    verdict: str = "insufficient_data"


def robustness_verdict(sharpes: List[float]) -> Dict[str, Any]:
    """Evaluate a series of out‑of‑sample Sharpe ratios against the robustness protocol.

    Parameters
    ----------
    sharpes: List[float]
        Sharpe ratios obtained from each walk‑forward window.

    Returns
    -------
    Dict[str, Any]
        Mapping containing the number of windows, the deflated Sharpe, the
        consistency fraction, a boolean ``is_robust`` flag, and a textual verdict.
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

    reasons = []
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


def _run_window(
    train: pd.Series,
    test: pd.Series,
    signals_fn: Callable[[pd.Series, pd.Series], pd.Series],
    equity_carry: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float]:
    """Execute a single walk‑forward window.

    The provided ``signals_fn`` is called with the training and testing price
    series and must return a ``pd.Series`` of signals aligned with ``test``.
    The backtest is then run on the test slice, and the resulting equity curve
    (if any) is returned together with updated equity carry for the next window.

    Parameters
    ----------
    train: pd.Series
        Historical price data used for training.
    test: pd.Series
        Out‑of‑sample price data used for testing.
    signals_fn: Callable[[pd.Series, pd.Series], pd.Series]
        Function that generates trading signals for the test period.
    equity_carry: float
        Starting equity for the backtest; may be updated with the ending equity
        of the current window.

    Returns
    -------
    Tuple[Dict[str, Any], List[Dict[str, Any]], float]
        * Window result dictionary (metrics or error information).
        * Equity‑curve list produced by the backtest (empty on error).
        * Updated equity carry for the subsequent window.
    """
    try:
        test_signals = signals_fn(train, test)
        metrics: BacktestMetrics = run_backtest(test_signals, test, initial_equity=equity_carry)
        new_carry = (
            metrics.equity_curve[-1]["equity"]
            if metrics.equity_curve
            else equity_carry
        )
        window_info = {
            KEY_START: str(test.index[0].date()),
            KEY_END: str(test.index[-1].date()),
            KEY_SHARPE: metrics.sharpe,
            KEY_MAX_DRAWDOWN: metrics.max_drawdown,
            KEY_TOTAL_RETURN: metrics.total_return,
            KEY_NUM_TRADES: metrics.num_trades,
        }
        return window_info, metrics.equity_curve, new_carry
    except Exception as e:
        error_info = {
            KEY_START: str(test.index[0].date()),
            KEY_END: str(test.index[-1].date()),
            KEY_ERROR: str(e),
        }
        return error_info, [], equity_carry


def _aggregate_averages(result: WalkForwardResult) -> None:
    """Populate ``result`` with average Sharpe and drawdown across successful windows.

    Parameters
    ----------
    result: WalkForwardResult
        The result object whose ``windows`` attribute contains per‑window dictionaries.
    """
    sharpe_vals = [w[KEY_SHARPE] for w in result.windows if KEY_SHARPE in w]
    drawdown_vals = [w[KEY_MAX_DRAWDOWN] for w in result.windows if KEY_MAX_DRAWDOWN in w]

    result.avg_sharpe = round(sum(sharpe_vals) / len(sharpe_vals), 4) if sharpe_vals else 0.0
    result.avg_drawdown = round(sum(drawdown_vals) / len(drawdown_vals), 4) if drawdown_vals else 0.0


def walk_forward(
    signals_fn: Callable[[pd.Series, pd.Series], pd.Series],
    prices: pd.Series,
    train_years: Optional[int] = None,
    test_months: Optional[int] = None,
    initial_equity: Optional[float] = None,
) -> WalkForwardResult:
    """Perform walk‑forward validation across the entire price history.

    The function iteratively slices ``prices`` into training and testing windows,
    invokes ``signals_fn`` to obtain trading signals for each test slice, and
    runs a backtest.  After all windows have been processed, average performance
    metrics are computed and a robustness verdict is attached.

    Parameters
    ----------
    signals_fn: Callable[[pd.Series, pd.Series], pd.Series]
        Callable that receives training and testing price series and returns a
        signal series for the test period.
    prices: pd.Series
        Full historical price series indexed by datetime.
    train_years: Optional[int]
        Number of years to use for each training window; defaults to ``TIMEFRAME_TRAIN``.
    test_months: Optional[int]
        Number of months to use for each testing window; defaults to ``TIMEFRAME_TEST``.
    initial_equity: Optional[float]
        Starting equity for the first backtest; defaults to ``MAX_EQUITY``.

    Returns
    -------
    WalkForwardResult
        Aggregated results, including per‑window metrics, combined equity curve,
        and robustness assessment.
    """
    train_bars = (train_years if train_years is not None else TIMEFRAME_TRAIN) * DAYS_PER_YEAR
    test_bars = (test_months if test_months is not None else TIMEFRAME_TEST) * DAYS_PER_MONTH

    result = WalkForwardResult()
    equity_carry = initial_equity if initial_equity is not None else MAX_EQUITY

    i = train_bars
    while i + test_bars <= len(prices):
        train_slice = prices.iloc[i - train_bars : i]
        test_slice = prices.iloc[i : i + test_bars]

        window_info, equity_curve, equity_carry = _run_window(
            train_slice,
            test_slice,
            signals_fn,
            equity_carry,
        )
        result.windows.append(window_info)
        result.combined_equity.extend(equity_curve)

        i += test_bars

    _aggregate_averages(result)

    verdict = robustness_verdict(
        [w[KEY_SHARPE] for w in result.windows if KEY_SHARPE in w]
    )
    result.n_windows = verdict["n_windows"]
    result.deflated_sharpe = verdict["deflated_sharpe"]
    result.consistency = verdict["consistency"]
    result.is_robust = verdict["is_robust"]
    result.verdict = verdict["verdict"]
    return result