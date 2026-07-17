"""
Historical stress testing — overlay a strategy's signals on known crisis periods.

Tests how a strategy would have performed during the most severe market dislocations,
revealing tail-risk exposure that standard backtests can understate when they
average across calm and turbulent regimes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, List

import pandas as pd

from app.backtest.engine import BacktestMetrics, run_backtest


@dataclass
class StressScenario:
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
    scenario: StressScenario
    # None if the price data doesn't cover this period
    metrics: BacktestMetrics | None
    period_covered: bool
    data_points: int


def _slice_series(series: pd.Series | None, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    """Vectorized slice of a Series using .loc; returns None if input is None."""
    if series is None:
        return None
    # .loc works for both DatetimeIndex and PeriodIndex; fallback to boolean mask if needed
    try:
        return series.loc[start:end]
    except Exception:
        mask = (series.index >= start) & (series.index <= end)
        return series.loc[mask]


def _validate_series(name: str, series: pd.Series | None, allow_none: bool = False) -> None:
    """Validate that an object is a pandas Series with a DatetimeIndex."""
    if series is None:
        if not allow_none:
            raise ValueError(f"'{name}' must be a pandas Series, not None.")
        return
    if not isinstance(series, pd.Series):
        raise ValueError(f"'{name}' must be a pandas Series, got {type(series)}.")
    if not isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError(f"'{name}' index must be a DatetimeIndex or PeriodIndex, got {type(series.index)}.")
    if series.empty:
        raise ValueError(f"'{name}' Series must contain at least one data point.")


def _validate_positive_number(name: str, value: float, allow_zero: bool = False) -> None:
    """Validate that a numeric value is positive (or non‑negative if allowed)."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number, got {type(value)}.")
    if allow_zero:
        if value < 0:
            raise ValueError(f"'{name}' must be non‑negative, got {value}.")
    else:
        if value <= 0:
            raise ValueError(f"'{name}' must be positive, got {value}.")


def _validate_percentage(name: str, value: float) -> None:
    """Validate that a percentage is between 0 and 1 inclusive."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number, got {type(value)}.")
    if not (0 <= value <= 1):
        raise ValueError(f"'{name}' must be between 0 and 1 inclusive, got {value}.")


def _validate_scenarios(scenarios: Iterable[StressScenario] | None) -> List[StressScenario]:
    """Validate and normalise the scenarios argument."""
    if scenarios is None:
        return STRESS_SCENARIOS
    if not isinstance(scenarios, Iterable):
        raise ValueError("scenarios must be an iterable of StressScenario objects.")
    validated: List[StressScenario] = []
    for s in scenarios:
        if not isinstance(s, StressScenario):
            raise ValueError(f"All items in scenarios must be StressScenario instances, got {type(s)}.")
        if s.start > s.end:
            raise ValueError(f"Scenario '{s.name}' has start date after end date.")
        validated.append(s)
    return validated


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
    Run the strategy through each stress scenario window.

    Only scenarios where the price series has ≥ 5 data points are evaluated;
    others return period_covered=False with metrics=None.
    """
    # Input validation
    _validate_series("signals", signals)
    _validate_series("prices", prices)
    _validate_series("opens", opens, allow_none=True)
    _validate_series("volume", volume, allow_none=True)
    _validate_positive_number("initial_equity", initial_equity)
    _validate_percentage("commission_pct", commission_pct)
    _validate_percentage("slippage_pct", slippage_pct)

    validated_scenarios = _validate_scenarios(scenarios)

    results: list[StressResult] = []

    # Convert once to pandas Timestamp for efficient comparison
    price_index = prices.index

    for scenario in validated_scenarios:
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


def stress_summary(results: list[StressResult]) -> dict:
    """
    Compact summary dict suitable for JSON serialisation.

    Returns per-scenario max_drawdown, total_return, and sharpe.
    Only includes scenarios where period_covered=True.
    """
    if not isinstance(results, list):
        raise ValueError("results must be a list of StressResult objects.")
    for idx, r in enumerate(results):
        if not isinstance(r, StressResult):
            raise ValueError(f"Item at index {idx} in results is not a StressResult, got {type(r)}.")

    out: dict = {}
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