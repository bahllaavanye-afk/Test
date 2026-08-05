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
    metrics: BacktestMetrics | None  # None if the price data doesn't cover this period
    period_covered: bool
    data_points: int


def _validate_series(series: pd.Series | None, name: str, allow_none: bool = False) -> None:
    """Validate that a pandas Series is provided (or optionally None)."""
    if series is None:
        if not allow_none:
            raise ValueError(f"'{name}' must be a pandas Series, not None.")
        return
    if not isinstance(series, pd.Series):
        raise ValueError(f"'{name}' must be a pandas Series, got {type(series).__name__}.")


def _validate_float(value: float, name: str, min_value: float = 0.0, max_value: float | None = None) -> None:
    """Validate that a numeric value is within the optional bounds."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number, got {type(value).__name__}.")
    if value < min_value:
        raise ValueError(f"'{name}' must be >= {min_value}, got {value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"'{name}' must be <= {max_value}, got {value}.")


def _validate_scenarios(scenarios: Iterable[StressScenario] | None) -> List[StressScenario]:
    """Validate that scenarios is an iterable of StressScenario objects."""
    if scenarios is None:
        return STRESS_SCENARIOS
    if not isinstance(scenarios, Iterable):
        raise ValueError("scenarios must be an iterable of StressScenario instances.")
    validated: List[StressScenario] = []
    for idx, s in enumerate(scenarios):
        if not isinstance(s, StressScenario):
            raise ValueError(f"scenarios[{idx}] is not a StressScenario instance.")
        validated.append(s)
    return validated


def run_stress_tests(
    signals: pd.Series,
    prices: pd.Series,
    opens: pd.Series | None = None,
    volume: pd.Series | None = None,
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    scenarios: list[StressScenario] | None = None,
) -> list[StressResult]:
    """
    Run the strategy through each stress scenario window.

    Only scenarios where the price series has ≥ MIN_DATA_POINTS data points are evaluated;
    others return period_covered=False with metrics=None.
    """
    # Input validation
    _validate_series(signals, "signals")
    _validate_series(prices, "prices")
    _validate_series(opens, "opens", allow_none=True)
    _validate_series(volume, "volume", allow_none=True)
    _validate_float(initial_equity, "initial_equity", min_value=0.0)
    _validate_float(commission_pct, "commission_pct", min_value=0.0, max_value=1.0)
    _validate_float(slippage_pct, "slippage_pct", min_value=0.0, max_value=1.0)

    validated_scenarios = _validate_scenarios(scenarios)

    results: list[StressResult] = []
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


def stress_summary(results: list[StressResult]) -> dict:
    """
    Compact summary dict suitable for JSON serialisation.

    Returns per‑scenario max_drawdown, total_return, and sharpe.
    Only includes scenarios where period_covered=True.
    """
    if not isinstance(results, list):
        raise ValueError("results must be a list of StressResult objects.")
    for idx, r in enumerate(results):
        if not isinstance(r, StressResult):
            raise ValueError(f"results[{idx}] is not a StressResult instance.")

    out: dict = {}
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