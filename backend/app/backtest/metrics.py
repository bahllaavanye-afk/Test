from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
    try:
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
            try:
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
            except Exception as e:
                logger.error(f"Error calculating information ratio: {str(e)}")

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
            try:
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
            except Exception as e:
                logger.error(f"Error calculating trade-level stats: {str(e)}")
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

        return BacktestMetrics(
            total_return_pct,
            annual_return_pct,
            sharpe,
            sortino,
            calmar,
            max_drawdown_pct,
            avg_drawdown_pct,
            max_dd_duration,
            total_trades,
            win_rate,
            avg_win_pct,
            avg_loss_pct,
            profit_factor,
            var_95,
            cvar_95,
            information_ratio,
            best_month_pct,
            worst_month_pct,
            recovery_factor
        )
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise
    except TypeError as e:
        logger.error(f"TypeError: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise