"""
Comprehensive backtest performance metrics.
All metrics follow institutional conventions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestMetrics:
    # Returns
    total_return_pct: float
    annual_return_pct: float

    # Risk-adjusted
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
    var_95: float           # 95% 1-day VaR (negative number = loss)
    cvar_95: float          # Expected Shortfall at 95%

    # Information ratio vs benchmark
    information_ratio: float

    # Extra
    best_month_pct: float
    worst_month_pct: float
    recovery_factor: float  # total_return / max_drawdown


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
) -> BacktestMetrics:
    """
    Compute all performance metrics from an equity curve.

    equity_curve : pd.Series indexed by date, values are portfolio equity ($).
    trades       : optional DataFrame with columns [pnl, hold_days].
    benchmark    : optional Series of benchmark equity (same index or overlapping).
    """
    if equity_curve is None or len(equity_curve) < 2:
        raise ValueError("equity_curve must have at least 2 data points")

    equity = equity_curve.dropna().astype(float)
    daily_returns = equity.pct_change().dropna()

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------
    initial = float(equity.iloc[0])
    final = float(equity.iloc[-1])
    total_return = (final / initial) - 1.0
    total_return_pct = round(total_return * 100, 4)

    n_days = len(equity)
    years = n_days / 252.0
    annual_return = (final / initial) ** (1.0 / max(years, 1e-6)) - 1.0
    annual_return_pct = round(annual_return * 100, 4)

    # ------------------------------------------------------------------
    # Sharpe  (rf = 0)
    # ------------------------------------------------------------------
    daily_std = float(daily_returns.std())
    if daily_std > 0:
        sharpe = float(daily_returns.mean() / daily_std * np.sqrt(252))
    else:
        sharpe = 0.0

    # ------------------------------------------------------------------
    # Sortino  (downside deviation below 0)
    # ------------------------------------------------------------------
    downside = daily_returns[daily_returns < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.0
    if downside_std > 0:
        sortino = float(daily_returns.mean() / downside_std * np.sqrt(252))
    else:
        sortino = 0.0

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------
    rolling_max = equity.cummax()
    drawdown_series = (equity - rolling_max) / rolling_max  # <= 0

    max_drawdown = float(drawdown_series.min())       # most negative
    max_drawdown_pct = round(max_drawdown * 100, 4)
    avg_drawdown_pct = round(float(drawdown_series[drawdown_series < 0].mean()) * 100, 4) if (drawdown_series < 0).any() else 0.0

    # Max drawdown duration: longest streak of bars below the peak
    in_dd = (drawdown_series < 0).astype(int)
    if in_dd.any():
        # group consecutive drawdown bars
        groups = (in_dd != in_dd.shift()).cumsum()
        dd_lengths = in_dd.groupby(groups).sum()
        max_dd_duration = int(dd_lengths.max())
    else:
        max_dd_duration = 0

    # ------------------------------------------------------------------
    # Calmar
    # ------------------------------------------------------------------
    if max_drawdown != 0:
        calmar = round(annual_return / abs(max_drawdown), 4)
    else:
        calmar = 0.0

    # ------------------------------------------------------------------
    # Recovery factor
    # ------------------------------------------------------------------
    if max_drawdown != 0:
        recovery_factor = round(total_return / abs(max_drawdown), 4)
    else:
        recovery_factor = 0.0

    # ------------------------------------------------------------------
    # VaR and CVaR (95%)
    # ------------------------------------------------------------------
    ret_arr = daily_returns.values
    var_95 = float(np.percentile(ret_arr, 5))       # 5th percentile = 95% VaR
    cvar_mask = ret_arr <= var_95
    cvar_95 = float(ret_arr[cvar_mask].mean()) if cvar_mask.any() else var_95

    # ------------------------------------------------------------------
    # Information ratio vs benchmark
    # ------------------------------------------------------------------
    information_ratio = 0.0
    if benchmark is not None and len(benchmark) > 1:
        bm = benchmark.dropna().astype(float)
        # Align on common index
        common_idx = equity.index.intersection(bm.index)
        if len(common_idx) > 1:
            strat_ret = equity.loc[common_idx].pct_change().dropna()
            bm_ret = bm.loc[common_idx].pct_change().dropna()
            # Re-align after pct_change
            common2 = strat_ret.index.intersection(bm_ret.index)
            if len(common2) > 1:
                active_returns = strat_ret.loc[common2] - bm_ret.loc[common2]
                tracking_error = float(active_returns.std()) * np.sqrt(252)
                if tracking_error > 0:
                    information_ratio = round(
                        float(active_returns.mean()) * 252 / tracking_error, 4
                    )

    # ------------------------------------------------------------------
    # Monthly best/worst
    # ------------------------------------------------------------------
    if hasattr(equity.index, 'to_period'):
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
    # Trade-level stats (from trades DataFrame if provided)
    # ------------------------------------------------------------------
    total_trades = 0
    win_rate = 0.0
    avg_win_pct = 0.0
    avg_loss_pct = 0.0
    profit_factor = 0.0

    if trades is not None and len(trades) > 0 and "pnl" in trades.columns:
        pnl = trades["pnl"].dropna()
        total_trades = len(pnl)
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        win_rate = round(len(wins) / total_trades, 4) if total_trades > 0 else 0.0
        avg_win_pct = round(float(wins.mean()) * 100, 4) if len(wins) > 0 else 0.0
        avg_loss_pct = round(float(losses.mean()) * 100, 4) if len(losses) > 0 else 0.0
        sum_losses = float(losses.sum())
        if sum_losses != 0:
            profit_factor = round(float(wins.sum()) / abs(sum_losses), 4)
        else:
            profit_factor = float("inf") if len(wins) > 0 else 0.0
    else:
        # Infer rough trade stats from equity curve sign-changes in daily returns
        signs = np.sign(daily_returns.values)
        sign_changes = np.where(np.diff(signs) != 0)[0]
        total_trades = len(sign_changes)

        pos_returns = daily_returns[daily_returns > 0]
        neg_returns = daily_returns[daily_returns <= 0]
        n_total = len(daily_returns)
        win_rate = round(len(pos_returns) / n_total, 4) if n_total > 0 else 0.0
        avg_win_pct = round(float(pos_returns.mean()) * 100, 4) if len(pos_returns) > 0 else 0.0
        avg_loss_pct = round(float(neg_returns.mean()) * 100, 4) if len(neg_returns) > 0 else 0.0
        sum_losses = float(neg_returns.sum())
        if sum_losses != 0:
            profit_factor = round(float(pos_returns.sum()) / abs(sum_losses), 4)
        else:
            profit_factor = float("inf") if len(pos_returns) > 0 else 0.0

    return BacktestMetrics(
        total_return_pct=total_return_pct,
        annual_return_pct=annual_return_pct,
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        calmar=calmar,
        max_drawdown_pct=max_drawdown_pct,
        avg_drawdown_pct=avg_drawdown_pct,
        max_drawdown_duration_days=max_dd_duration,
        total_trades=total_trades,
        win_rate=win_rate,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        profit_factor=profit_factor,
        var_95=round(var_95, 6),
        cvar_95=round(cvar_95, 6),
        information_ratio=information_ratio,
        best_month_pct=best_month_pct,
        worst_month_pct=worst_month_pct,
        recovery_factor=recovery_factor,
    )


def compute_mae_mfe(trades: list[dict]) -> dict:
    """
    Maximum Adverse Excursion and Maximum Favorable Excursion.
    Each trade dict needs: entry_price, exit_price, side, high_during_trade, low_during_trade
    (uses exit_price as proxy for high/low if OHLCV not available).
    Returns: mae_mean, mae_p95, mfe_mean, mfe_p95, edge_ratio, efficiency
    """
    if not trades:
        return {}
    mae_list, mfe_list, eff_list = [], [], []
    for t in trades:
        entry = float(t.get("entry_price", 0) or 0)
        exit_p = float(t.get("exit_price", 0) or 0)
        side = t.get("side", "buy")
        high = float(t.get("high_during_trade", max(entry, exit_p)) or max(entry, exit_p))
        low = float(t.get("low_during_trade", min(entry, exit_p)) or min(entry, exit_p))
        if entry <= 0:
            continue
        if side == "buy":
            mae = max(0, (entry - low) / entry)
            mfe = max(0, (high - entry) / entry)
            realized = max(0, (exit_p - entry) / entry)
        else:
            mae = max(0, (high - entry) / entry)
            mfe = max(0, (entry - low) / entry)
            realized = max(0, (entry - exit_p) / entry)
        mae_list.append(mae)
        mfe_list.append(mfe)
        eff_list.append(realized / mfe if mfe > 0 else 0)
    if not mae_list:
        return {}
    return {
        "mae_mean": float(np.mean(mae_list)),
        "mae_p95": float(np.percentile(mae_list, 95)),
        "mfe_mean": float(np.mean(mfe_list)),
        "mfe_p95": float(np.percentile(mfe_list, 95)),
        "edge_ratio": float(np.mean(mfe_list) / np.mean(mae_list)) if np.mean(mae_list) > 0 else 0,
        "efficiency": float(np.mean(eff_list)),
    }


def compute_r_multiples(trades: list[dict]) -> dict:
    """
    R-multiple = realized_pnl / initial_risk.
    initial_risk = entry * stop_loss_pct if stop_loss not set, else abs(entry - stop_loss).
    trade dict needs: realized_pnl, entry_price, stop_loss_price (optional), quantity
    """
    if not trades:
        return {}
    r_multiples = []
    for t in trades:
        entry = float(t.get("entry_price", 0) or 0)
        pnl = float(t.get("realized_pnl", 0) or 0)
        qty = float(t.get("quantity", 1) or 1)
        stop = t.get("stop_loss_price")
        if entry <= 0 or qty <= 0:
            continue
        if stop and float(stop) > 0:
            risk_per_share = abs(entry - float(stop))
        else:
            risk_per_share = entry * 0.02  # default 2% ATR proxy
        initial_risk = risk_per_share * qty
        if initial_risk > 0:
            r_multiples.append(pnl / initial_risk)
    if not r_multiples:
        return {}
    arr = np.array(r_multiples)
    winners = arr[arr > 0]
    losers = arr[arr < 0]
    return {
        "expectancy_R": float(np.mean(arr)),
        "positive_expectancy": bool(np.mean(arr) > 0),
        "avg_winner_R": float(np.mean(winners)) if len(winners) > 0 else 0.0,
        "avg_loser_R": float(np.mean(losers)) if len(losers) > 0 else 0.0,
        "win_rate": float(len(winners) / len(arr)),
        "profit_factor": float(np.sum(winners) / abs(np.sum(losers))) if len(losers) > 0 and np.sum(losers) != 0 else float("inf"),
        "r_multiple_p25": float(np.percentile(arr, 25)),
        "r_multiple_p75": float(np.percentile(arr, 75)),
        "r_multiple_distribution": [round(float(v), 3) for v in np.clip(arr, -10, 10).tolist()[:200]],
    }


def compute_position_metrics(trades: list[dict]) -> dict:
    """
    Position handling quality metrics.
    trade dict needs: hold_seconds, quantity, realized_pnl, entry_price,
                      filled_qty (optional), ordered_qty (optional)
    """
    if not trades:
        return {}
    hold_hours, sizes, fill_rates = [], [], []
    entry_slippages, exit_slippages = [], []
    for t in trades:
        if t.get("hold_seconds"):
            hold_hours.append(float(t["hold_seconds"]) / 3600)
        entry = float(t.get("entry_price", 0) or 0)
        qty = float(t.get("quantity", 0) or 0)
        if entry > 0 and qty > 0:
            sizes.append(entry * qty)
        filled = t.get("filled_qty")
        ordered = t.get("ordered_qty")
        if filled and ordered and float(ordered) > 0:
            fill_rates.append(float(filled) / float(ordered))
        entry_slip = t.get("entry_slippage_bps")
        exit_slip = t.get("exit_slippage_bps")
        if entry_slip is not None:
            entry_slippages.append(float(entry_slip))
        if exit_slip is not None:
            exit_slippages.append(float(exit_slip))
    result = {}
    if hold_hours:
        result.update({
            "avg_hold_hours": float(np.mean(hold_hours)),
            "median_hold_hours": float(np.median(hold_hours)),
            "max_hold_hours": float(np.max(hold_hours)),
        })
    if fill_rates:
        result["partial_fill_rate"] = float(np.mean([r < 1.0 for r in fill_rates]))
    if entry_slippages:
        result["avg_slippage_entry_bps"] = float(np.mean(entry_slippages))
    if exit_slippages:
        result["avg_slippage_exit_bps"] = float(np.mean(exit_slippages))
    return result
