from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetricComputationError(Exception):
    """Custom exception for errors occurring during metric computation."""
    pass


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
        try:
            initial = float(equity.iloc[0])
            final = float(equity.iloc[-1])
            total_return = (final / initial) - 1.0
            total_return_pct = round(total_return * 100, 4)

            n_days = len(equity)
            years = n_days / 252.0
            annual_return = (final / initial) ** (1.0 / max(years, 1e-6)) - 1.0
            annual_return_pct = round(annual_return * 100, 4)
        except (IndexError, ZeroDivisionError) as e:
            logger.error("Error computing returns", exc_info=True)
            raise MetricComputationError("Failed to compute returns") from e

        # ------------------------------------------------------------------
        # Sharpe  (rf = 0)
        # ------------------------------------------------------------------
        try:
            daily_std = float(daily_returns.std())
            sharpe = (
                float(daily_returns.mean() / daily_std * np.sqrt(252))
                if daily_std > 0
                else 0.0
            )
        except Exception as e:
            logger.error("Error computing Sharpe ratio", exc_info=True)
            raise MetricComputationError("Failed to compute Sharpe ratio") from e

        # ------------------------------------------------------------------
        # Sortino  (downside deviation below 0)
        # ------------------------------------------------------------------
        try:
            downside = daily_returns[daily_returns < 0]
            downside_std = float(downside.std()) if len(downside) > 1 else 0.0
            sortino = (
                float(daily_returns.mean() / downside_std * np.sqrt(252))
                if downside_std > 0
                else 0.0
            )
        except Exception as e:
            logger.error("Error computing Sortino ratio", exc_info=True)
            raise MetricComputationError("Failed to compute Sortino ratio") from e

        # ------------------------------------------------------------------
        # Drawdown
        # ------------------------------------------------------------------
        try:
            rolling_max = equity.cummax()
            drawdown_series = (equity - rolling_max) / rolling_max  # <= 0

            max_drawdown = float(drawdown_series.min())  # most negative
            max_drawdown_pct = round(max_drawdown * 100, 4)

            if (drawdown_series < 0).any():
                avg_drawdown_pct = round(
                    float(drawdown_series[drawdown_series < 0].mean()) * 100, 4
                )
            else:
                avg_drawdown_pct = 0.0

            # Max drawdown duration: longest streak of bars below the peak
            in_dd = (drawdown_series < 0).astype(int)
            if in_dd.any():
                groups = (in_dd != in_dd.shift()).cumsum()
                dd_lengths = in_dd.groupby(groups).sum()
                max_dd_duration = int(dd_lengths.max())
            else:
                max_dd_duration = 0
        except Exception as e:
            logger.error("Error computing drawdown metrics", exc_info=True)
            raise MetricComputationError("Failed to compute drawdown metrics") from e

        # ------------------------------------------------------------------
        # Calmar
        # ------------------------------------------------------------------
        try:
            calmar = round(annual_return / abs(max_drawdown), 4) if max_drawdown != 0 else 0.0
        except Exception as e:
            logger.error("Error computing Calmar ratio", exc_info=True)
            raise MetricComputationError("Failed to compute Calmar ratio") from e

        # ------------------------------------------------------------------
        # Recovery factor
        # ------------------------------------------------------------------
        try:
            recovery_factor = (
                round(total_return / abs(max_drawdown), 4) if max_drawdown != 0 else 0.0
            )
        except Exception as e:
            logger.error("Error computing recovery factor", exc_info=True)
            raise MetricComputationError("Failed to compute recovery factor") from e

        # ------------------------------------------------------------------
        # VaR and CVaR (95%)
        # ------------------------------------------------------------------
        try:
            ret_arr = daily_returns.values
            var_95 = float(np.percentile(ret_arr, 5))  # 5th percentile = 95% VaR
            cvar_mask = ret_arr <= var_95
            cvar_95 = float(ret_arr[cvar_mask].mean()) if cvar_mask.any() else var_95
        except Exception as e:
            logger.error("Error computing VaR/CVaR", exc_info=True)
            raise MetricComputationError("Failed to compute VaR/CVaR") from e

        # ------------------------------------------------------------------
        # Information ratio vs benchmark
        # ------------------------------------------------------------------
        information_ratio = 0.0
        if benchmark is not None and len(benchmark) > 1:
            try:
                bm = benchmark.dropna().astype(float)
                common_idx = equity.index.intersection(bm.index)
                if len(common_idx) > 1:
                    strat_ret = equity.loc[common_idx].pct_change().dropna()
                    bm_ret = bm.loc[common_idx].pct_change().dropna()
                    common2 = strat_ret.index.intersection(bm_ret.index)
                    if len(common2) > 1:
                        active_returns = strat_ret.loc[common2] - bm_ret.loc[common2]
                        tracking_error = float(active_returns.std()) * np.sqrt(252)
                        if tracking_error > 0:
                            information_ratio = round(
                                float(active_returns.mean()) * 252 / tracking_error, 4
                            )
            except Exception as e:
                logger.error("Error calculating information ratio", exc_info=True)

        # ------------------------------------------------------------------
        # Monthly best/worst
        # ------------------------------------------------------------------
        try:
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
        except Exception as e:
            logger.error("Error computing monthly best/worst", exc_info=True)
            raise MetricComputationError("Failed to compute monthly best/worst") from e

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

                loss_sum = float(losses.sum())
                profit_factor = (
                    round(float(wins.sum()) / abs(loss_sum), 4)
                    if loss_sum != 0
                    else np.inf
                )
            except Exception as e:
                logger.error("Error calculating trade statistics", exc_info=True)

        # ------------------------------------------------------------------
        # Assemble results
        # ------------------------------------------------------------------
        return BacktestMetrics(
            total_return_pct=total_return_pct,
            annual_return_pct=annual_return_pct,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            max_drawdown_pct=max_drawdown_pct,
            avg_drawdown_pct=avg_drawdown_pct,
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
    except MetricComputationError:
        # Already logged; re‑raise to caller
        raise
    except Exception as e:
        logger.exception("Unexpected error during metric computation")
        raise MetricComputationError("Unexpected error during metric computation") from e