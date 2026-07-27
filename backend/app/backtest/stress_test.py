"""
Historical stress testing — overlay a strategy's signals on known crisis periods.

This module provides utilities to evaluate a trading strategy under historically
significant market dislocations. By restricting the back‑test to predefined
stress windows, users can expose tail‑risk characteristics that are often hidden
by full‑period back‑tests which blend calm and turbulent regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.backtest.engine import BacktestMetrics, run_backtest


@dataclass
class StressScenario:
    """
    Definition of a stress test window.

    Attributes
    ----------
    name: str
        Unique identifier used as a dictionary key in results.
    label: str
        Short human‑readable label for visualisations.
    start: date
        Inclusive start date of the stress window.
    end: date
        Inclusive end date of the stress window.
    description: str
        Brief narrative describing the market event.
    """
    name: str
    label: str          # short label for charts
    start: date
    end: date
    description: str


# Canonical crisis windows used by institutional risk teams
STRESS_SCENARIOS: list[StressScenario] = [
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
    Outcome of a single stress scenario evaluation.

    Attributes
    ----------
    scenario: StressScenario
        The scenario that was evaluated.
    metrics: BacktestMetrics | None
        Back‑test performance metrics; ``None`` when the period could not be
        evaluated (e.g., insufficient price data).
    period_covered: bool
        ``True`` if the price series contained enough data points for the window.
    data_points: int
        Number of price observations used in the back‑test.
    """
    scenario: StressScenario
    # None if the price data doesn't cover this period
    metrics: BacktestMetrics | None
    period_covered: bool
    data_points: int


def _slice_series(
    series: pd.Series | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | None:
    """
    Slice a pandas Series to the inclusive ``[start, end]`` window.

    Parameters
    ----------
    series: pd.Series | None
        Series to be sliced; ``None`` is propagated unchanged.
    start: pd.Timestamp
        Inclusive start of the slice.
    end: pd.Timestamp
        Inclusive end of the slice.

    Returns
    -------
    pd.Series | None
        The sliced series or ``None`` if the input was ``None``.
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
    opens: pd.Series | None = None,
    volume: pd.Series | None = None,
    initial_equity: float = 100_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    scenarios: list[StressScenario] | None = None,
) -> list[StressResult]:
    """
    Execute back‑tests for each defined stress scenario.

    The function extracts the portion of the input time series that falls within
    each scenario window and runs a standard back‑test using :func:`run_backtest`.
    Scenarios with fewer than five price points are marked as not covered.

    Parameters
    ----------
    signals : pd.Series
        Strategy signal series (e.g., binary or continuous exposure values).
    prices : pd.Series
        Price series used for P&L calculation.
    opens : pd.Series | None, optional
        Optional open price series; if ``None`` the back‑test will use ``prices``.
    volume : pd.Series | None, optional
        Optional volume series for liquidity‑adjusted metrics.
    initial_equity : float, default 100_000.0
        Starting capital for the back‑test.
    commission_pct : float, default 0.001
        Commission cost expressed as a fraction of trade value.
    slippage_pct : float, default 0.0005
        Slippage cost expressed as a fraction of trade value.
    scenarios : list[StressScenario] | None, optional
        Custom list of stress windows; defaults to the built‑in
        ``STRESS_SCENARIOS`` constant.

    Returns
    -------
    list[StressResult]
        One result object per scenario, containing metrics and coverage flags.
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    results: list[StressResult] = []

    # Convert once to pandas Timestamp for efficient comparison
    price_index: pd.Index = prices.index

    for scenario in scenarios:
        start_ts: pd.Timestamp = pd.Timestamp(scenario.start)
        end_ts: pd.Timestamp = pd.Timestamp(scenario.end)

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

        s_signals: pd.Series | None = _slice_series(signals, start_ts, end_ts)
        s_prices: pd.Series | None = _slice_series(prices, start_ts, end_ts)
        s_opens: pd.Series | None = (
            _slice_series(opens, start_ts, end_ts) if opens is not None else None
        )
        s_volume: pd.Series | None = (
            _slice_series(volume, start_ts, end_ts) if volume is not None else None
        )

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

        metrics: BacktestMetrics = run_backtest(
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


def stress_summary(results: list[StressResult]) -> dict[str, dict]:
    """
    Produce a JSON‑serialisable summary of stress test outcomes.

    For each scenario the function reports coverage information and, when
    applicable, key performance indicators such as total return, maximum draw‑down,
    Sharpe ratio, win rate and trade count.

    Parameters
    ----------
    results : list[StressResult]
        List returned by :func:`run_stress_tests`.

    Returns
    -------
    dict[str, dict]
        Mapping from scenario ``name`` to a dictionary containing human‑readable
        fields and numeric metrics (rounded where appropriate).
    """
    out: dict[str, dict] = {}
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