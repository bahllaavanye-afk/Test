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
# Unit Tests for Edge Cases
# ----------------------------------------------------------------------
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta


class DummyMetrics:
    """Simple stand‑in for BacktestMetrics with required attributes."""
    def __init__(self, total_return=0.05, max_drawdown=-0.1, sharpe=1.2, win_rate=0.6, num_trades=10):
        self.total_return = total_return
        self.max_drawdown = max_drawdown
        self.sharpe = sharpe
        self.win_rate = win_rate
        self.num_trades = num_trades


class TestStressTestEdgeCases(unittest.TestCase):
    def setUp(self):
        # Base daily index for tests
        self.base_date = datetime(2020, 1, 1)
        self.dates = pd.date_range(self.base_date, periods=30, freq="D")

    def test_no_overlap_scenario(self):
        """Scenario window does not intersect price data → period_covered=False."""
        prices = pd.Series([100 + i for i in range(30)], index=self.dates)
        signals = pd.Series([1] * 30, index=self.dates)

        # Define a scenario completely outside the price range
        outside_scenario = StressScenario(
            name="outside",
            label="Outside",
            start=date(1990, 1, 1),
            end=date(1990, 1, 10),
            description="No overlap with price data",
        )

        results = run_stress_tests(signals, prices, scenarios=[outside_scenario])
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertFalse(res.period_covered)
        self.assertIsNone(res.metrics)
        self.assertEqual(res.data_points, 0)

    def test_exact_boundary_five_points(self):
        """Exactly 5 data points in the window should be processed."""
        # Create price series that spans exactly 5 days within the scenario
        scenario_start = self.base_date + timedelta(days=5)
        scenario_end = scenario_start + timedelta(days=4)  # 5 days total
        dates = pd.date_range(scenario_start, scenario_end, freq="D")
        prices = pd.Series([100 + i for i in range(5)], index=dates)
        signals = pd.Series([1] * 5, index=dates)

        exact_scenario = StressScenario(
            name="exact_five",
            label="Exact Five",
            start=scenario_start.date(),
            end=scenario_end.date(),
            description="Exactly five data points",
        )

        with patch(
            "backend.app.backtest.stress_test.run_backtest",
            return_value=DummyMetrics(),
        ):
            results = run_stress_tests(signals, prices, scenarios=[exact_scenario])

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertTrue(res.period_covered)
        self.assertIsInstance(res.metrics, DummyMetrics)
        self.assertEqual(res.data_points, 5)

    def test_insufficient_data_points(self):
        """Fewer than 5 points should result in period_covered=False."""
        scenario_start = self.base_date + timedelta(days=1)
        scenario_end = scenario_start + timedelta(days=2)  # 3 days total
        dates = pd.date_range(scenario_start, scenario_end, freq="D")
        prices = pd.Series([100, 101, 102], index=dates)
        signals = pd.Series([1, 1, 1], index=dates)

        short_scenario = StressScenario(
            name="short",
            label="Short",
            start=scenario_start.date(),
            end=scenario_end.date(),
            description="Less than five data points",
        )

        results = run_stress_tests(signals, prices, scenarios=[short_scenario])
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertFalse(res.period_covered)
        self.assertIsNone(res.metrics)
        self.assertEqual(res.data_points, 3)

    def test_stress_summary_representation(self):
        """stress_summary should correctly represent covered and uncovered scenarios."""
        # Covered scenario
        covered_scenario = StressScenario(
            name="covered",
            label="Covered",
            start=date(2020, 1, 1),
            end=date(2020, 1, 10),
            description="Covered scenario",
        )
        # Uncovered scenario
        uncovered_scenario = StressScenario(
            name="uncovered",
            label="Uncovered",
            start=date(1999, 1, 1),
            end=date(1999, 1, 5),
            description="Uncovered scenario",
        )

        # Dummy result objects
        covered_result = StressResult(
            scenario=covered_scenario,
            metrics=DummyMetrics(total_return=0.1, max_drawdown=-0.05, sharpe=1.5, win_rate=0.7, num_trades=20),
            period_covered=True,
            data_points=8,
        )
        uncovered_result = StressResult(
            scenario=uncovered_scenario,
            metrics=None,
            period_covered=False,
            data_points=0,
        )

        summary = stress_summary([covered_result, uncovered_result])
        self.assertIn("covered", summary["covered"])
        self.assertTrue(summary["covered"]["covered"])
        self.assertEqual(summary["covered"]["total_return_pct"], 10.0)
        self.assertIn("uncovered", summary)
        self.assertFalse(summary["uncovered"]["covered"])
        self.assertEqual(summary["uncovered"]["label"], "Uncovered")


if __name__ == "__main__":
    unittest.main()