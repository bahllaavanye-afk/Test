"""
Historical stress testing — overlay a strategy's signals on known crisis periods.

Tests how a strategy would have performed during the most severe market dislocations,
revealing tail-risk exposure that standard backtests can understate when they
average across calm and turbulent regimes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

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
    metrics: BacktestMetrics | None  # None if the price data doesn't cover this period
    period_covered: bool
    data_points: int


def _slice_series(
    series: pd.Series | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | None:
    """Slice a Series using .loc; returns None if input is None."""
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
    scenarios: list[StressScenario] | None = None,
) -> list[StressResult]:
    """
    Run the strategy through each stress scenario window.

    Only scenarios where the price series has ≥ MIN_DATA_POINTS data points are evaluated;
    others return period_covered=False with metrics=None.
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    results: list[StressResult] = []
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


def stress_summary(results: list[StressResult]) -> dict:
    """
    Compact summary dict suitable for JSON serialisation.

    Returns per‑scenario max_drawdown, total_return, and sharpe.
    Only includes scenarios where period_covered=True.
    """
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

# --------------------------------------------------------------------------- #
# Unit tests for edge‑case handling
# --------------------------------------------------------------------------- #

import unittest
from types import SimpleNamespace
from unittest.mock import patch


class TestStressTestingEdgeCases(unittest.TestCase):
    """Targeted tests for boundary conditions in stress testing."""

    def setUp(self) -> None:
        # Create a simple time index spanning 10 days
        self.dates = pd.date_range(start="2021-01-01", periods=10, freq="D")
        # Signals and prices are identical for simplicity
        self.signals = pd.Series([1] * 10, index=self.dates)
        self.prices = pd.Series([100 + i for i in range(10)], index=self.dates)

        # Dummy metrics object with required attributes
        self.dummy_metrics = SimpleNamespace(
            total_return=0.05,
            max_drawdown=-0.02,
            sharpe=1.2,
            win_rate=0.6,
            num_trades=5,
        )

    @patch("app.backtest.stress_test.run_backtest")
    def test_scenario_with_insufficient_data_points(self, mock_run_backtest):
        """Scenario where sliced price series has fewer than MIN_DATA_POINTS."""
        mock_run_backtest.return_value = self.dummy_metrics

        # Define a scenario that only overlaps the first 3 days
        short_scenario = StressScenario(
            name="short",
            label="Short",
            start=date(2021, 1, 1),
            end=date(2021, 1, 3),
            description="Only three data points",
        )

        results = run_stress_tests(
            signals=self.signals,
            prices=self.prices,
            scenarios=[short_scenario],
        )
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertFalse(res.period_covered)
        self.assertIsNone(res.metrics)
        self.assertEqual(res.data_points, 3)
        # Ensure run_backtest was never called due to insufficient data
        mock_run_backtest.assert_not_called()

    @patch("app.backtest.stress_test.run_backtest")
    def test_scenario_with_exact_min_data_points(self, mock_run_backtest):
        """Scenario where sliced price series has exactly MIN_DATA_POINTS."""
        mock_run_backtest.return_value = self.dummy_metrics

        # Define a scenario that overlaps exactly 5 days
        exact_scenario = StressScenario(
            name="exact",
            label="Exact",
            start=date(2021, 1, 1),
            end=date(2021, 1, 5),
            description="Exactly MIN_DATA_POINTS",
        )

        results = run_stress_tests(
            signals=self.signals,
            prices=self.prices,
            scenarios=[exact_scenario],
        )
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertTrue(res.period_covered)
        self.assertIsNotNone(res.metrics)
        self.assertEqual(res.data_points, MIN_DATA_POINTS)
        mock_run_backtest.assert_called_once()

    def test_slice_series_none_input(self):
        """_slice_series should return None when the input series is None."""
        start = pd.Timestamp("2021-01-01")
        end = pd.Timestamp("2021-01-05")
        self.assertIsNone(_slice_series(None, start, end))

    def test_slice_series_out_of_range(self):
        """When the slice range is outside the series index, an empty Series should be returned."""
        start = pd.Timestamp("2020-12-01")
        end = pd.Timestamp("2020-12-05")
        result = _slice_series(self.prices, start, end)
        # Result should be a Series with length 0 (no matching dates)
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()