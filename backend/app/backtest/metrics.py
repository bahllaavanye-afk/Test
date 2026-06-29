"""
Optimized backtest performance metrics.
All metrics follow institutional conventions.

The most expensive part of the original implementation was the
drawdown duration calculation, which used pandas `groupby` on a
generated series.  This version replaces it with a pure‑numpy run‑length
algorithm, reducing overhead and avoiding the creation of intermediate
objects.  Minor refactoring also removes redundant calculations and
adds early‑exit guards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class BacktestMetrics:
    # Returns
    total_return_pct: float
    annual_return_pct: float

    # Risk‑adjusted
    sharpe: float           # annualized, rf=0
    sortino: float          # downside deviation
    calmar: float           # annual_return / max_drawdown

    # Drawdown
    max_drawdown_pct: float
    avg_drawdown_pct: float
    max_drawdown_duration_days: int

    # Trading stats
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float    # sum(wins) / sum(losses)

    # Tail risk
    var_95: float           # 95% 1‑day VaR (negative number = loss)
    cvar_95: float          # Expected Shortfall at 95%

    # Information ratio vs benchmark
    information_ratio: float

    # Extra
    best_month_pct: float
    worst_month_pct: float
    recovery_factor: float  # total_return / max_drawdown


def _max_consecutive_true(arr: np.ndarray) -> int:
    """
    Return the length of the longest run of consecutive `True` (or 1) values.
    Uses a pure‑numpy implementation to avoid pandas overhead.
    """
    if arr.size == 0:
        return 0
    # Ensure binary integer array (0/1)
    arr = arr.astype(np.int8)
    # Pad with a zero to capture a run that ends at the last element
    padded = np.concatenate([arr, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size == 0:
        return 0
    return int((ends - starts).max())


def compute_metrics(
    equity_curve: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series] = None,
) -> BacktestMetrics:
    """
    Compute a full suite of performance metrics from an equity curve.

    Parameters
    ----------
    equity_curve : pd.Series
        Indexed by date, values are portfolio equity ($).
    trades : pd.DataFrame, optional
        Must contain at least a ``pnl`` column. If omitted, trade‑level
        statistics are approximated from daily returns.
    benchmark : pd.Series, optional
        Series of benchmark equity (same index or overlapping).

    Returns
    -------
    BacktestMetrics
        Dataclass containing all computed metrics.
    """
    if equity_curve is None or len(equity_curve) < 2:
        raise ValueError("equity_curve must have at least 2 data points")

    # ------------------------------------------------------------------
    # Clean data
    # ------------------------------------------------------------------
    equity = equity_curve.dropna().astype(float)
    daily_returns = equity.pct_change().dropna()
    if daily_returns.empty:
        raise ValueError("Equity curve must contain at least one non‑zero return")

    # Cache common aggregates
    daily_mean = float(daily_returns.mean())
    daily_std = float(daily_returns.std())
    n_days = len(equity)
    years = n_days / 252.0 if n_days > 0 else 0.0

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------
    initial = float(equity.iloc[0])
    final = float(equity.iloc[-1])
    total_return = (final / initial) - 1.0
    total_return_pct = round(total_return * 100, 4)

    annual_return = (final / initial) ** (1.0 / max(years, 1e-6)) - 1.0
    annual_return_pct = round(annual_return * 100, 4)

    # ------------------------------------------------------------------
    # Sharpe (rf = 0)
    # ------------------------------------------------------------------
    sharpe = (
        float(daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # Sortino (downside deviation)
    # ------------------------------------------------------------------
    downside = daily_returns[daily_returns < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = (
        float(daily_mean / downside_std * np.sqrt(252))
        if downside_std > 0
        else 0.0
    )

    # ------------------------------------------------------------------
    # Drawdown metrics
    # ------------------------------------------------------------------
    rolling_max = equity.cummax()
    drawdown_series = (equity - rolling_max) / rolling_max  # <= 0

    max_drawdown = float(drawdown_series.min())  # most negative
    max_drawdown_pct = round(max_drawdown * 100, 4)

    if (drawdown_series < 0).any():
        avg_dd = float(drawdown_series[drawdown_series < 0].mean())
        avg_drawdown_pct = round(avg_dd * 100, 4)
    else:
        avg_drawdown_pct = 0.0

    # Max drawdown duration: longest consecutive period below the peak
    in_dd = (drawdown_series < 0).values.astype(np.int8)
    max_dd_duration = _max_consecutive_true(in_dd)
    max_drawdown_duration_days = int(max_dd_duration)

    # ------------------------------------------------------------------
    # Calmar & Recovery factor
    # ------------------------------------------------------------------
    if max_drawdown != 0:
        calmar = round(annual_return / abs(max_drawdown), 4)
        recovery_factor = round(total_return / abs(max_drawdown), 4)
    else:
        calmar = 0.0
        recovery_factor = 0.0

    # ------------------------------------------------------------------
    # VaR / CVaR (95%)
    # ------------------------------------------------------------------
    ret_arr = daily_returns.values
    var_95 = float(np.percentile(ret_arr, 5))  # 5th percentile = 95% VaR
    cvar_mask = ret_arr <= var_95
    cvar_95 = float(ret_arr[cvar_mask].mean()) if cvar_mask.any() else var_95

    # ------------------------------------------------------------------
    # Information ratio vs benchmark
    # ------------------------------------------------------------------
    information_ratio = 0.0
    if benchmark is not None and len(benchmark) > 1:
        bm = benchmark.dropna().astype(float)
        common_idx = equity.index.intersection(bm.index)
        if len(common_idx) > 1:
            strat_ret = equity.loc[common_idx].pct_change().dropna()
            bm_ret = bm.loc[common_idx].pct_change().dropna()
            common2 = strat_ret.index.intersection(bm_ret.index)
            if len(common2) > 1:
                active = strat_ret.loc[common2] - bm_ret.loc[common2]
                tracking_error = float(active.std()) * np.sqrt(252)
                if tracking_error > 0:
                    information_ratio = round(
                        float(active.mean()) * 252 / tracking_error, 4
                    )

    # ------------------------------------------------------------------
    # Monthly best / worst
    # ------------------------------------------------------------------
    if hasattr(equity.index, "to_period"):
        monthly = equity.resample("ME").last()
        monthly_returns = monthly.pct_change().dropna()
    else:
        monthly_returns = pd.Series(dtype=float)

    if len(monthly_returns) > 0:
        best_month_pct = round(float(monthly_returns.max()) * 100, 4)
        worst_month_pct = round(float(monthly_returns.min()) * 100, 4)
    else:
        best_month_pct = 0.0
        worst_month_pct = 0.0

    # ------------------------------------------------------------------
    # Trade‑level statistics
    # ------------------------------------------------------------------
    total_trades = 0
    win_rate = 0.0
    avg_win_pct = 0.0
    avg_loss_pct = 0.0
    profit_factor = 0.0

    if trades is not None and len(trades) > 0 and "pnl" in trades.columns:
        pnl = trades["pnl"].dropna().astype(float)
        total_trades = len(pnl)
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]

        win_rate = round(len(wins) / total_trades, 4) if total_trades else 0.0
        avg_win_pct = round(float(wins.mean()) * 100, 4) if len(wins) else 0.0
        avg_loss_pct = round(float(losses.mean()) * 100, 4) if len(losses) else 0.0

        sum_losses = float(losses.sum())
        if sum_losses != 0:
            profit_factor = round(float(wins.sum()) / abs(sum_losses), 4)
        else:
            profit_factor = float("inf") if len(wins) else 0.0
    else:
        # Approximate trade stats from daily returns
        total_trades = len(daily_returns)
        pos = daily_returns[daily_returns > 0]
        neg = daily_returns[daily_returns <= 0]

        win_rate = round(len(pos) / total_trades, 4) if total_trades else 0.0
        avg_win_pct = round(float(pos.mean()) * 100, 4) if len(pos) else 0.0
        avg_loss_pct = round(float(neg.mean()) * 100, 4) if len(neg) else 0.0

        sum_losses = float(neg.sum())
        if sum_losses != 0:
            profit_factor = round(float(pos.sum()) / abs(sum_losses), 4)
        else:
            profit_factor = float("inf") if len(pos) else 0.0

    return BacktestMetrics(
        total_return_pct=total_return_pct,
        annual_return_pct=annual_return_pct,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown_pct=max_drawdown_pct,
        avg_drawdown_pct=avg_drawdown_pct,
        max_drawdown_duration_days=max_drawdown_duration_days,
        total_trades=total_trades,
        win_rate=win_rate,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        profit_factor=profit_factor,
        var_95=var_95,
        cvar_95=cvar_95,
        information_ratio=information_ratio,
        best_month_pct=best_month_pct,
        worst_month_pct=worst_month_pct,
        recovery_factor=recovery_factor,
    )