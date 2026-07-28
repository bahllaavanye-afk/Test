"""
Historical stress testing — overlay a strategy's signals on known crisis periods.

Tests how a strategy would have performed during the most severe market dislocations,
revealing tail‑risk exposure that standard backtests can understate when they
average across calm and turbulent regimes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from app.backtest.engine import BacktestMetrics, run_backtest


@dataclass
class StressScenario:
    """
    Definition of a stress testing window.

    Attributes
    ----------
    name: str
        Unique identifier for the scenario (used as a dictionary key).
    label: str
        Short human‑readable label for charts and reports.
    start: date
        Inclusive start date of the stress window.
    end: date
        Inclusive end date of the stress window.
    description: str
        Brief description of the market event.
    """
    name: str
    label: str          # short label for charts
    start: date
    end: date
    description: str


# Canonical crisis windows used by institutional risk teams
STRESS_SCENARIOS: List[StressScenario] = [
    StressScenario(
        "gfc",
        "GFC 2008",
        date(2008, 9, 1),
        date(2009, 3, 31),
        "Global Financial Crisis: Lehman collapse through S&P trough",
    ),
    StressScenario(
        "euro_crisis",
        "Euro Crisis 2011",
        date(2011, 7, 1),
        date(2011, 10, 31),
        "European sovereign debt crisis peak: S&P −20% in 3 months",
    ),
    StressScenario(
        "china_flash",
        "China Flash 2015",
        date(2015, 8, 17),
        date(2015, 9, 30),
        "China yuan devaluation + flash crash: S&P −12% in 6 days",
    ),
    StressScenario(
        "vol_spike_2018",
        "Vol Spike Feb-18",
        date(2018, 1, 26),
        date(2018, 2, 28),
        "VIX inverse ETN collapse: S&P −10% in 2 weeks",
    ),
    StressScenario(
        "covid_crash",
        "COVID Crash 2020",
        date(2020, 2, 20),
        date(2020, 3, 23),
        "COVID-19 panic: S&P −34% in 23 trading days (fastest in history)",
    ),
    StressScenario(
        "rate_hike_2022",
        "Rate Hikes 2022",
        date(2022, 1, 3),
        date(2022, 12, 31),
        "Fed tightening cycle: S&P −19.4%, Nasdaq −33%, bonds −15%",
    ),
    StressScenario(
        "svb_2023",
        "SVB Crisis 2023",
        date(2023, 3, 6),
        date(2023, 3, 31),
        "Silicon Valley Bank collapse and banking sector contagion",
    ),
]


@dataclass
class StressResult:
    """
    Result of running a backtest over a single stress scenario.

    Attributes
    ----------
    scenario: StressScenario
        The scenario that was evaluated.
    metrics: BacktestMetrics | None
        Backtest performance metrics, or ``None`` if the scenario could not be
        evaluated (e.g., insufficient price data).
    period_covered: bool
        ``True`` if the price series covered the scenario window sufficiently.
    data_points: int
        Number of price data points used for the backtest.
    """
    scenario: StressScenario
    metrics: BacktestMetrics | None
    period_covered: bool
    data_points: int


def _slice_series(
    series: Optional[pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Optional[pd.Series]:
    """
    Slice a pandas Series to the inclusive ``[start, end]`` window.

    Parameters
    ----------
    series: pd.Series | None
        The series to slice. ``None`` is returned unchanged.
    start: pd.Timestamp
        Start timestamp of the slice (inclusive).
    end: pd.Timestamp
        End timestamp of the slice (inclusive).

    Returns
    -------
    pd.Series | None
        The sliced series, or ``None`` if the input series is ``None``.
    """
    if series is None:
        return None
    # .loc works for both DatetimeIndex and PeriodIndex; fallback to boolean mask if needed
    try:
        return series.loc[start:end]
    except Exception:
        mask = (series.index >= start) & (series.index <= end)
        return series.loc[mask]


def run_stress_tests(
    signals: pd.Series,
    prices: pd.Series,
    opens: Optional[pd.Series] = None,
    volume: Optional[pd.Series] = None,
    initial_equity: float = 100_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    scenarios: Optional[List[StressScenario]] = None,
) -> List[StressResult]:
    """
    Execute the strategy backtest across each defined stress scenario.

    Only scenarios where the price series contains at least five data points are
    evaluated; other windows are reported with ``period_covered=False`` and
    ``metrics=None``.

    Parameters
    ----------
    signals : pd.Series
        Time‑indexed series of binary or continuous signals.
    prices : pd.Series
        Time‑indexed series of closing prices.
    opens : pd.Series | None, optional
        Time‑indexed series of opening prices, if required by the strategy.
    volume : pd.Series | None, optional
        Time‑indexed series of traded volume.
    initial_equity : float, default 100_000.0
        Starting capital for the backtest.
    commission_pct : float, default 0.001
        Commission as a proportion of trade value.
    slippage_pct : float, default 0.0005
        Slippage as a proportion of trade value.
    scenarios : list[StressScenario] | None, optional
        Custom list of scenarios to evaluate; defaults to the canonical set.

    Returns
    -------
    list[StressResult]
        One result object per scenario, preserving order.
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    results: List[StressResult] = []

    # Convert once to pandas Timestamp for efficient comparison
    price_index = prices.index

    for scenario in scenarios:
        start_ts = pd.Timestamp(scenario.start)
        end_ts = pd.Timestamp(scenario.end)

        # Fast check: if the scenario window does not intersect the price index, skip early
        if not ((price_index >= start_ts) & (price_index <= end_ts)).any():
            results.append(
                StressResult(
                    scenario=scenario,
                    metrics=None,
                    period_covered=False,
                    data_points=0,
                )
            )
            continue

        s_signals = _slice_series(signals, start_ts, end_ts)
        s_prices = _slice_series(prices, start_ts, end_ts)
        s_opens = _slice_series(opens, start_ts, end_ts) if opens is not None else None
        s_volume = _slice_series(volume, start_ts, end_ts) if volume is not None else None

        if s_prices is None or len(s_prices) < 5:
            results.append(
                StressResult(
                    scenario=scenario,
                    metrics=None,
                    period_covered=False,
                    data_points=len(s_prices) if s_prices is not None else 0,
                )
            )
            continue

        metrics = run_backtest(
            signals=s_signals,
            prices=s_prices,
            opens=s_opens,
            volume=s_volume,
            initial_equity=initial_equity,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )

        results.append(
            StressResult(
                scenario=scenario,
                metrics=metrics,
                period_covered=True,
                data_points=len(s_prices),
            )
        )

    return results


def stress_summary(results: List[StressResult]) -> Dict[str, dict]:
    """
    Produce a compact JSON‑serialisable summary of stress test outcomes.

    For each scenario the dictionary contains coverage information, basic
    performance metrics (total return, max drawdown, Sharpe ratio, win rate, and
    number of trades), and metadata such as the label and description.

    Parameters
    ----------
    results : list[StressResult]
        List of results returned by :func:`run_stress_tests`.

    Returns
    -------
    dict
        Mapping from scenario name to a dictionary of summary fields.
    """
    out: Dict[str, dict] = {}
    for r in results:
        if not r.period_covered or r.metrics is None:
            out[r.scenario.name] = {
                "covered": False,
                "label": r.scenario.label,
                "description": r.scenario.description,
            }
        else:
            out[r.scenario.name] = {
                "covered": True,
                "label": r.scenario.label,
                "description": r.scenario.description,
                "data_points": r.data_points,
                "total_return_pct": round(r.metrics.total_return * 100, 2),
                "max_drawdown_pct": round(r.metrics.max_drawdown * 100, 2),
                "sharpe": r.metrics.sharpe,
                "win_rate": r.metrics.win_rate,
                "num_trades": r.metrics.num_trades,
            }
    return out