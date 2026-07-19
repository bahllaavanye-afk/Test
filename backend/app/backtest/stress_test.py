"""
Historical stress testing — overlay a strategy's signals on known crisis periods.

Tests how a strategy would have performed during the most severe market dislocations,
revealing tail-risk exposure that standard backtests can understate when they
average across calm and turbulent regimes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

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
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    results: list[StressResult] = []

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


def stress_summary(results: list[StressResult]) -> dict:
    """
    Compact summary dict suitable for JSON serialisation.

    Returns per-scenario max_drawdown, total_return, and sharpe.
    Only includes scenarios where period_covered=True.
    """
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


# ----------------------------------------------------------------------
# Unit tests for edge‑case handling of run_stress_tests
# ----------------------------------------------------------------------
def _create_series(start: str, periods: int, freq: str = "B", value: float = 1.0) -> pd.Series:
    """Utility to generate a simple numeric series with a DatetimeIndex."""
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    return pd.Series([value] * periods, index=idx)


def test_exact_boundary_five_points():
    """
    Scenario where the price series contains exactly 5 data points
    within the stress window. The function should treat this as a
    valid period (period_covered=True).
    """
    # Define a scenario covering 5 business days
    scenario = StressScenario(
        name="exact_five",
        label="Exact Five Days",
        start=date(2021, 1, 4),   # Monday
        end=date(2021, 1, 8),     # Friday
        description="Exactly five trading days",
    )
    # Create price and signal series that align with the scenario dates
    prices = _create_series(start="2021-01-04", periods=5)
    signals = _create_series(start="2021-01-04", periods=5, value=0.0)

    results = run_stress_tests(
        signals=signals,
        prices=prices,
        scenarios=[scenario],
    )
    assert len(results) == 1
    res = results[0]
    assert res.period_covered is True
    assert res.data_points == 5
    # Metrics should be produced (non‑None) when the backtest runs
    assert res.metrics is not None


def test_insufficient_data_points():
    """
    Scenario where the price series contains fewer than 5 data points
    within the stress window. The function should return period_covered=False.
    """
    scenario = StressScenario(
        name="insufficient",
        label="Insufficient Data",
        start=date(2021, 2, 1),
        end=date(2021, 2, 10),
        description="Less than five data points",
    )
    # Only 4 business days in the range
    prices = _create_series(start="2021-02-01", periods=4)
    signals = _create_series(start="2021-02-01", periods=4, value=0.0)

    results = run_stress_tests(
        signals=signals,
        prices=prices,
        scenarios=[scenario],
    )
    assert len(results) == 1
    res = results[0]
    assert res.period_covered is False
    assert res.data_points == 4
    assert res.metrics is None


def test_no_intersection_between_price_and_scenario():
    """
    Scenario where the price index does not intersect the stress window at all.
    The function should quickly return period_covered=False with zero data points.
    """
    scenario = StressScenario(
        name="no_intersect",
        label="No Intersection",
        start=date(1999, 1, 1),
        end=date(1999, 1, 31),
        description="Price series outside scenario range",
    )
    # Price series well after the scenario dates
    prices = _create_series(start="2022-01-01", periods=10)
    signals = _create_series(start="2022-01-01", periods=10, value=0.0)

    results = run_stress_tests(
        signals=signals,
        prices=prices,
        scenarios=[scenario],
    )
    assert len(results) == 1
    res = results[0]
    assert res.period_covered is False
    assert res.data_points == 0
    assert res.metrics is None