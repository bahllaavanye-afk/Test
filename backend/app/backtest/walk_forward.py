"""Walk-forward validation: train on N years, test on M months, roll forward."""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field

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
    windows: list[dict] = field(default_factory=list)
    avg_sharpe: float = 0.0
    avg_drawdown: float = 0.0
    combined_equity: list[dict] = field(default_factory=list)
    # Overfit gate (populated by walk_forward()):
    n_windows: int = 0
    deflated_sharpe: float = 0.0   # DSR over the window Sharpes (multiple‑testing haircut)
    consistency: float = 0.0       # fraction of windows with Sharpe ≥ MIN_OOS_SHARPE
    is_robust: bool = False        # passes the full protocol → safe to promote
    verdict: str = "insufficient_data"


def robustness_verdict(sharpes: list[float]) -> dict:
    """Grade a walk‑forward's per‑window Sharpes against the documented protocol.

    Pure + side‑effect free so it can be unit‑tested directly. A strategy is
    ``robust`` only if it has enough OOS windows, an average and a majority of
    windows clearing the per‑window bar, AND a positive Deflated Sharpe (so a
    handful of lucky windows can't carry it past the multiple‑testing haircut).
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


def _apply_signal_filters(
    train: pd.Series,
    test: pd.Series,
    raw_signals: pd.Series,
    max_holding_days: int = 10,
) -> pd.Series:
    """Tighten entry conditions and add simple confirmation / exit filters.

    - **Trend confirmation**: Use 20‑day and 50‑day simple moving averages
      computed on the concatenated train+test series. Long entries are kept only
      when the 20‑day SMA is above the 50‑day SMA; short entries require the
      opposite.
    - **Persistence filter**: Require the same signal direction to appear for
      at least two consecutive days before a trade is opened.
    - **Maximum holding period**: After an entry, force an exit (signal = 0)
      after ``max_holding_days`` to avoid excessively long positions.

    The function returns a signal series aligned with ``test`` where any
    disallowed signal is replaced by 0.
    """
    # Align raw signals to test index and fill missing values with 0
    signals = raw_signals.reindex(test.index).fillna(0)

    # --- Trend confirmation -------------------------------------------------
    combined = pd.concat([train, test])
    sma20 = combined.rolling(window=20, min_periods=1).mean()
    sma50 = combined.rolling(window=50, min_periods=1).mean()
    sma20_test = sma20.reindex(test.index)
    sma50_test = sma50.reindex(test.index)

    trend_ok = ((signals > 0) & (sma20_test > sma50_test)) | (
        (signals < 0) & (sma20_test < sma50_test)
    ) | (signals == 0)

    # --- Persistence filter -------------------------------------------------
    persistence = (signals == signals.shift(1)) | (signals == 0)

    # Combine filters
    allowed = trend_ok & persistence

    filtered = signals.where(allowed, other=0)

    # --- Maximum holding period ---------------------------------------------
    # Convert to positions: 1 for long, -1 for short, 0 for flat
    position = filtered.copy()
    holding_counter = 0
    for idx in range(len(position)):
        sig = position.iloc[idx]
        if sig != 0:
            if holding_counter == 0:
                # New entry
                holding_counter = 1
            else:
                holding_counter += 1
            if holding_counter > max_holding_days:
                # Force exit
                position.iloc[idx] = 0
                holding_counter = 0
        else:
            holding_counter = 0
    return position


def _run_window(
    train: pd.Series,
    test: pd.Series,
    signals_fn,
    equity_carry: float,
) -> tuple[dict, list[dict], float]:
    """Execute a single walk‑forward window.

    Returns a tuple of:
    - window result dict (either metrics or error)
    - equity curve produced by the backtest (empty if error)
    - updated equity carry for the next window
    """
    try:
        raw_signals = signals_fn(train, test)
        signals = _apply_signal_filters(train, test, raw_signals)

        metrics: BacktestMetrics = run_backtest(signals, test, initial_equity=equity_carry)

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
    """Compute average Sharpe and drawdown from the collected windows."""
    sharpe_vals = [w[KEY_SHARPE] for w in result.windows if KEY_SHARPE in w]
    drawdown_vals = [w[KEY_MAX_DRAWDOWN] for w in result.windows if KEY_MAX_DRAWDOWN in w]

    result.avg_sharpe = round(sum(sharpe_vals) / len(sharpe_vals), 4) if sharpe_vals else 0.0
    result.avg_drawdown = round(sum(drawdown_vals) / len(drawdown_vals), 4) if drawdown_vals else 0.0


def walk_forward(
    signals_fn,               # callable(train_df, test_df) -> pd.Series of signals on test_df
    prices: pd.Series,
    train_years: int | None = None,
    test_months: int | None = None,
    initial_equity: float | None = None,
) -> WalkForwardResult:
    """
    Rolls a train/test window across entire history.
    `signals_fn` receives (train_prices, test_prices) and must return signals for the test period only.
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

    # Compute average metrics
    _aggregate_averages(result)

    # Overfit gate: grade the OOS window distribution (DSR + consistency).
    verdict = robustness_verdict(
        [w[KEY_SHARPE] for w in result.windows if KEY_SHARPE in w]
    )
    result.n_windows = verdict["n_windows"]
    result.deflated_sharpe = verdict["deflated_sharpe"]
    result.consistency = verdict["consistency"]
    result.is_robust = verdict["is_robust"]
    result.verdict = verdict["verdict"]
    return result