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
    arr = arr.astype(np.int8)
    padded = np.concatenate([arr, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size == 0:
        return 0
    return int((ends - starts).max())


def _calculate_returns(equity: pd.Series) -> tuple[float, float]:
    initial = float(equity.iloc[0])
    final = float(equity.iloc[-1])
    total_return = (final / initial) - 1.0
    total_return_pct = round(total_return * 100, 4)

    n_days = len(equity)
    years = n_days / 252.0 if n_days > 0 else 0.0
    annual_return = (final / initial) ** (1.0 / max(years, 1e-6)) - 1.0
    annual_return_pct = round(annual_return * 100, 4)

    return total_return_pct, annual_return_pct


def _calculate_sharpe_sortino(daily_returns: pd.Series) -> tuple[float, float]:
    daily_mean = float(daily_returns.mean())
    daily_std = float(daily_returns.std())
    sharpe = float(daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

    downside = daily_returns[daily_returns < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = (
        float(daily_mean / downside_std * np.sqrt(252))
        if downside_std > 0
        else 0.0
    )
    return sharpe, sortino


def _calculate_drawdown_metrics(equity: pd.Series) -> tuple[float, float, int]:
    rolling_max = equity.cummax()
    drawdown_series = (equity - rolling_max) / rolling_max  # <= 0

    max_drawdown = float(drawdown_series.min())
    max_drawdown_pct = round(max_drawdown * 100, 4)

    if (drawdown_series < 0).any():
        avg_dd = float(drawdown_series[drawdown_series < 0].mean())
        avg_drawdown_pct = round(avg_dd * 100, 4)
    else:
        avg_drawdown_pct = 0.0

    in_dd = (drawdown_series < 0).values.astype(np.int8)
    max_dd_duration = _max_consecutive_true(in_dd)
    max_drawdown_duration_days = int(max_dd_duration)

    return max_drawdown_pct, avg_drawdown_pct, max_drawdown_duration_days


def _calculate_calmar_recovery(
    annual_return_pct: float,
    total_return_pct: float,
    max_drawdown_pct: float,
) -> tuple[float, float]:
    max_drawdown = max_drawdown_pct / 100.0
    if max_drawdown != 0:
        calmar = round((annual_return_pct / 100.0) / abs(max_drawdown), 4)
        recovery_factor = round((total_return_pct / 100.0) / abs(max_drawdown), 4)
    else:
        calmar = 0.0
        recovery_factor = 0.0
    return calmar, recovery_factor


def _calculate_var_cvar(daily_returns: pd.Series) -> tuple[float, float]:
    ret_arr = daily_returns.values
    var_95 = float(np.percentile(ret_arr, 5))  # 5th percentile = 95% VaR
    cvar_mask = ret_arr <= var_95
    cvar_95 = float(ret_arr[cvar_mask].mean()) if cvar_mask.any() else var_95
    return var_95, cvar_95


def _calculate_information_ratio(
    equity: pd.Series,
    benchmark: Optional[pd.Series],
) -> float:
    if benchmark is None or len(benchmark) <= 1:
        return 0.0

    bm = benchmark.dropna().astype(float)
    common_idx = equity.index.intersection(bm.index)
    if len(common_idx) <= 1:
        return 0.0

    strat_ret = equity.loc[common_idx].pct_change().dropna()
    bm_ret = bm.loc[common_idx].pct_change().dropna()
    common2 = strat_ret.index.intersection(bm_ret.index)
    if len(common2) <= 1:
        return 0.0

    active = strat_ret.loc[common2] - bm_ret.loc[common2]
    tracking_error = float(active.std()) * np.sqrt(252)
    if tracking_error == 0:
        return 0.0
    return round(float(active.mean()) * 252 / tracking_error, 4)


def _calculate_monthly_best_worst(equity: pd.Series) -> tuple[float, float]:
    if not hasattr(equity.index, "to_period"):
        return 0.0, 0.0
    monthly = equity.resample("ME").last()
    monthly_returns = monthly.pct_change().dropna()
    if len(monthly_returns) == 0:
        return 0.0, 0.0
    best_month_pct = round(float(monthly_returns.max()) * 100, 4)
    worst_month_pct = round(float(monthly_returns.min()) * 100, 4)
    return best_month_pct, worst_month_pct


def _calculate_trade_stats(
    trades: Optional[pd.DataFrame],
    daily_returns: pd.Series,
    initial_equity: float,
) -> tuple[int, float, float, float, float]:
    """
    Derive trade‑level statistics.  If a trades DataFrame is supplied,
    use its ``pnl`` column; otherwise approximate from daily returns.
    """
    if trades is not None and "pnl" in trades.columns:
        pnl_series = trades["pnl"].astype(float)
        total_trades = int(len(pnl_series))
        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]
    else:
        # Approximate: treat each daily return as a “trade”
        pnl_series = daily_returns * initial_equity
        total_trades = int(len(pnl_series))
        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]

    win_rate = round(len(wins) / total_trades, 4) if total_trades > 0 else 0.0
    avg_win_pct = round(wins.mean() / initial_equity * 100, 4) if not wins.empty else 0.0
    avg_loss_pct = round(losses.mean() / initial_equity * 100, 4) if not losses.empty else 0.0
    profit_factor = (
        round(wins.sum() / abs(losses.sum()), 4)
        if not losses.empty and losses.sum() != 0
        else 0.0
    )
    return total_trades, win_rate, avg_win_pct, avg_loss_pct, profit_factor


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

    # Clean data
    equity = equity_curve.dropna().astype(float)
    daily_returns = equity.pct_change().dropna()
    if daily_returns.empty:
        raise ValueError("Equity curve must contain at least one non‑zero return")

    # Returns
    total_return_pct, annual_return_pct = _calculate_returns(equity)

    # Sharpe & Sortino
    sharpe, sortino = _calculate_sharpe_sortino(daily_returns)

    # Drawdown metrics
    max_dd_pct, avg_dd_pct, max_dd_duration = _calculate_drawdown_metrics(equity)

    # Calmar & Recovery factor
    calmar, recovery_factor = _calculate_calmar_recovery(
        annual_return_pct, total_return_pct, max_dd_pct
    )

    # VaR / CVaR (95%)
    var_95, cvar_95 = _calculate_var_cvar(daily_returns)

    # Information ratio vs benchmark
    information_ratio = _calculate_information_ratio(equity, benchmark)

    # Monthly best / worst
    best_month_pct, worst_month_pct = _calculate_monthly_best_worst(equity)

    # Trade‑level statistics
    initial_equity = float(equity.iloc[0])
    (
        total_trades,
        win_rate,
        avg_win_pct,
        avg_loss_pct,
        profit_factor,
    ) = _calculate_trade_stats(trades, daily_returns, initial_equity)

    return BacktestMetrics(
        total_return_pct=total_return_pct,
        annual_return_pct=annual_return_pct,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown_pct=max_dd_pct,
        avg_drawdown_pct=avg_dd_pct,
        max_drawdown_duration_days=max_dd_duration,
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