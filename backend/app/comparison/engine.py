"""
Strategy Comparison Engine: run manual vs ML-enhanced strategy on same period,
compare against benchmarks, compute statistical significance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict

import pandas as pd
from scipy import stats

from app.backtest.engine import run_backtest, BacktestMetrics
from app.comparison.benchmarks import fetch_benchmark_curves, get_benchmark_stats
from app.utils.logging import logger


@dataclass
class ComparisonResult:
    """
    Container for the results of a strategy comparison.

    Attributes
    ----------
    strategy_name: str
        Name of the strategy being compared.
    symbol: str
        Trading symbol (e.g., ticker) used in the backtest.
    interval: str
        Data interval (e.g., "1h", "1d").
    start_date: date
        Start date of the backtest period.
    end_date: date
        End date of the backtest period.
    manual: BacktestMetrics | None
        Metrics from the manual (baseline) strategy.
    ml_enhanced: BacktestMetrics | None
        Metrics from the ML‑enhanced strategy.
    benchmark_curves: Dict[str, Any]
        Benchmark equity curves fetched for the period.
    benchmark_stats: Dict[str, Any]
        Statistical summary of the benchmarks.
    ml_improvement_sharpe: float
        Difference in Sharpe ratio (ML – manual).
    t_statistic: float
        t‑statistic from the two‑sample t‑test comparing returns.
    p_value: float
        p‑value from the t‑test.
    is_significant: bool
        Whether the p‑value indicates statistical significance (< 0.05).
    winner: str
        Identifier of the winning strategy: "ml", "manual", or "neither".
    """

    strategy_name: str
    symbol: str
    interval: str
    start_date: date
    end_date: date
    manual: BacktestMetrics | None = None
    ml_enhanced: BacktestMetrics | None = None
    benchmark_curves: Dict[str, Any] = field(default_factory=dict)
    benchmark_stats: Dict[str, Any] = field(default_factory=dict)
    ml_improvement_sharpe: float = 0.0
    t_statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    winner: str = "neither"


class StrategyComparisonEngine:
    async def run_comparison(
        self,
        manual_signals: pd.Series,
        ml_signals: pd.Series,
        prices: pd.Series,
        strategy_name: str,
        symbol: str,
        interval: str,
        start_date: date,
        end_date: date,
        initial_equity: float = 100_000,
    ) -> ComparisonResult:
        """
        Execute a side‑by‑side comparison between a manual strategy and its
        machine‑learning‑enhanced counterpart.

        The method runs backtests for both signal sets, fetches benchmark data,
        calculates return series, performs a two‑sample t‑test, and assembles a
        :class:`ComparisonResult` summarising the outcomes.

        Parameters
        ----------
        manual_signals : pd.Series
            Signal series for the baseline (manual) strategy.
        ml_signals : pd.Series
            Signal series for the ML‑enhanced strategy.
        prices : pd.Series
            Historical price series used for the backtests.
        strategy_name : str
            Human‑readable name of the strategy.
        symbol : str
            Trading symbol associated with the data.
        interval : str
            Data granularity (e.g., "1h", "1d").
        start_date : date
            Start date of the backtest period.
        end_date : date
            End date of the backtest period.
        initial_equity : float, default 100_000
            Starting capital for each backtest.

        Returns
        -------
        ComparisonResult
            Populated result object containing metrics, benchmark information,
            statistical test outcomes, and the identified winner.
        """
        manual_metrics = run_backtest(manual_signals, prices, initial_equity)
        ml_metrics = run_backtest(ml_signals, prices, initial_equity)

        benchmark_curves = await fetch_benchmark_curves(start_date, end_date)
        benchmark_stats = get_bbenchmark_stats()

        manual_eq = pd.Series([e["equity"] for e in manual_metrics.equity_curve])
        ml_eq = pd.Series([e["equity"] for e in ml_metrics.equity_curve])
        manual_ret = manual_eq.pct_change().dropna()
        ml_ret = ml_eq.pct_change().dropna()

        min_len = min(len(manual_ret), len(ml_ret))
        if min_len > 10:
            t_stat, p_val = stats.ttest_ind(ml_ret.iloc[:min_len], manual_ret.iloc[:min_len])
        else:
            t_stat, p_val = 0.0, 1.0

        improvement = ml_metrics.sharpe - manual_metrics.sharpe
        winner = "ml" if ml_metrics.sharpe > manual_metrics.sharpe else "manual"
        if abs(improvement) < 0.1:
            winner = "neither"

        logger.info(
            "Comparison complete",
            strategy=strategy_name,
            manual_sharpe=manual_metrics.sharpe,
            ml_sharpe=ml_metrics.sharpe,
            p_value=round(p_val, 4),
        )

        return ComparisonResult(
            strategy_name=strategy_name,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            manual=manual_metrics,
            ml_enhanced=ml_metrics,
            benchmark_curves=benchmark_curves,
            benchmark_stats=benchmark_stats,
            ml_improvement_sharpe=round(improvement, 4),
            t_statistic=round(float(t_stat), 4),
            p_value=round(float(p_val), 6),
            is_significant=p_val < 0.05,
            winner=winner,
        )