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

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_INITIAL_EQUITY: float = 100_000.0
DEFAULT_COMMISSION_PCT: float = 0.001
DEFAULT_SLIPPAGE_PCT: float = 0.0005
MIN_DATA_POINTS: int = 5
ZERO_DATA_POINTS: int = 0

# Percentage conversion constants for summary output
PCT_MULTIPLIER: float = 100.0
PCT_PRECISION: int = 2

# Dictionary keys used in the JSON‑serialisable summary
KEY_COVERED = "covered"
KEY_LABEL = "label"
KEY_DESCRIPTION = "description"
KEY_DATA_POINTS = "data_points"
KEY_TOTAL_RETURN_PCT = "total_return_pct"
KEY_MAX_DRAWDOWN_PCT = "max_drawdown_pct"
KEY_SHARPE = "sharpe"
KEY_WIN_RATE = "win_rate"
KEY_NUM_TRADES = "num_trades"


@dataclass
class StressScenario:
    """Container describing a single stress‑testing window.

    Attributes
    ----------
    name: str
        Unique identifier used as the dictionary key in summary output.
    label: str
        Short, human‑readable label for charts and UI.
    start: date
        Inclusive start date of the stress period.
    end: date
        Inclusive end date of the stress period.
    description: str
        Longer description of the market event.
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
    """Result of applying a strategy to a single stress scenario.

    Attributes
    ----------
    scenario: StressScenario
        The scenario definition.
    metrics: BacktestMetrics | None
        Backtest metrics for the scenario; ``None`` when insufficient data.
    period_covered: bool
        ``True`` if the price series covered the scenario window with enough points.
    data_points: int
        Number of price data points used (or ``0`` when not covered).
    """


    scenario: StressScenario
    metrics: BacktestMetrics | None  # None if the price data doesn't cover this period
    period_covered: bool
    data_points: int


def _slice_series(
    series: pd.Series | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | None:
    """Return a slice of ``series`` between ``start`` and ``end`` inclusive.

    If ``series`` is ``None`` the function returns ``None``. The slice is performed
    using ``.loc`` for label‑based indexing; if that fails (e.g., non‑datetime index),
    a boolean mask fallback is applied.

    Parameters
    ----------
    series: pd.Series | None
        The time‑series to slice.
    start: pd.Timestamp
        Inclusive start timestamp.
    end: pd.Timestamp
        Inclusive end timestamp.

    Returns
    -------
    pd.Series | None
        The sliced series or ``None`` when the input is ``None``.
    """
    if series is None:
        return None
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
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    scenarios: List[StressScenario] | None = None,
) -> List[StressResult]:
    """
    Execute backtests for each defined stress scenario.

    The function iterates over ``scenarios`` (defaulting to :data:`STRESS_SCENARIOS`),
    extracts the relevant slices of ``signals`` and ``prices`` (and optional ``opens``
    and ``volume``), and runs :func:`app.backtest.engine.run_backtest` when the slice
    contains at least :data:`MIN_DATA_POINTS` observations. Scenarios without enough
    data are marked as not covered.

    Parameters
    ----------
    signals : pd.Series
        Strategy signal series aligned with ``prices``.
    prices : pd.Series
        Price series used for the backtest.
    opens : pd.Series | None, optional
        Optional open price series.
    volume : pd.Series | None, optional
        Optional volume series.
    initial_equity : float, optional
        Starting equity for each backtest.
    commission_pct : float, optional
        Commission as a proportion of trade value.
    slippage_pct : float, optional
        Slippage as a proportion of trade value.
    scenarios : List[StressScenario] | None, optional
        Custom list of scenarios; if ``None`` the predefined list is used.

    Returns
    -------
    List[StressResult]
        A list containing the result for each scenario.
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    results: List[StressResult] = []
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
                    data_points=ZERO_DATA_POINTS,
                )
            )
            continue

        s_signals = _slice_series(signals, start_ts, end_ts)
        s_prices = _slice_series(prices, start_ts, end_ts)
        s_opens = _slice_series(opens, start_ts, end_ts) if opens is not None else None
        s_volume = _slice_series(volume, start_ts, end_ts) if volume is not None else None

        if s_prices is None or len(s_prices) < MIN_DATA_POINTS:
            results.append(
                StressResult(
                    scenario=scenario,
                    metrics=None,
                    period_covered=False,
                    data_points=len(s_prices) if s_prices is not None else ZERO_DATA_POINTS,
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


def stress_summary(results: List[StressResult]) -> Dict[str, Dict]:
    """
    Produce a compact, JSON‑serialisable summary of stress‑test results.

    For each scenario the summary includes coverage information and, when covered,
    key performance metrics such as total return, maximum drawdown, Sharpe ratio,
    win rate, and number of trades.

    Parameters
    ----------
    results : List[StressResult]
        The list returned by :func:`run_stress_tests`.

    Returns
    -------
    Dict[str, Dict]
        Mapping from scenario name to a dictionary of summary fields.
    """
    out: Dict[str, Dict] = {}
    for r in results:
        if not r.period_covered or r.metrics is None:
            out[r.scenario.name] = {
                KEY_COVERED: False,
                KEY_LABEL: r.scenario.label,
                KEY_DESCRIPTION: r.scenario.description,
            }
        else:
            out[r.scenario.name] = {
                KEY_COVERED: True,
                KEY_LABEL: r.scenario.label,
                KEY_DESCRIPTION: r.scenario.description,
                KEY_DATA_POINTS: r.data_points,
                KEY_TOTAL_RETURN_PCT: round(r.metrics.total_return * PCT_MULTIPLIER, PCT_PRECISION),
                KEY_MAX_DRAWDOWN_PCT: round(r.metrics.max_drawdown * PCT_MULTIPLIER, PCT_PRECISION),
                KEY_SHARPE: r.metrics.sharpe,
                KEY_WIN_RATE: r.metrics.win_rate,
                KEY_NUM_TRADES: r.metrics.num_trades,
            }
    return out


__all__ = [
    "StressScenario",
    "StressResult",
    "STRESS_SCENARIOS",
    "run_stress_tests",
    "stress_summary",
]