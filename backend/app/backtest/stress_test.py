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


# --------------------------------------------------------------------------- #
# Unit tests for edge cases
# --------------------------------------------------------------------------- #

import pytest
from unittest.mock import patch


def _dummy_metrics():
    """Create a minimal BacktestMetrics instance for mocking."""
    return BacktestMetrics(
        total_return=0.05,
        max_drawdown=-0.02,
        sharpe=1.1,
        win_rate=0.55,
        num_trades=4,
    )


def test_slice_series_none_input():
    """_slice_series should return None when the input series is None."""
    result = _slice_series(None, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31"))
    assert result is None


def test_slice_series_non_datetime_index():
    """_slice_series should correctly slice a Series with a non‑datetime index using a mask."""
    data = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])
    # Provide timestamps that correspond to integer index values after conversion
    start = pd.Timestamp("1970-01-01")  # maps to 0
    end = pd.Timestamp("1970-01-05")    # maps to 4
    result = _slice_series(data, start, end)
    pd.testing.assert_series_equal(result, data)


def test_run_stress_tests_boundary_min_points():
    """
    Scenario with exactly 5 data points should be evaluated (period_covered=True).
    """
    dates = pd.date_range(start="2022-01-01", periods=5, freq="D")
    prices = pd.Series([100, 101, 102, 103, 104], index=dates)
    signals = pd.Series([0, 1, 0, -1, 0], index=dates)

    # Create a custom scenario that exactly matches the dates above
    scenario = StressScenario(
        name="boundary_test",
        label="Boundary Test",
        start=date(2022, 1, 1),
        end=date(2022, 1, 5),
        description="Exactly five data points",
    )

    with patch("app.backtest.engine.run_backtest", return_value=_dummy_metrics()) as mock_bt:
        results = run_stress_tests(
            signals=signals,
            prices=prices,
            scenarios=[scenario],
        )
    assert len(results) == 1
    res = results[0]
    assert res.period_covered is True
    assert res.metrics is not None
    assert res.data_points == 5
    mock_bt.assert_called_once()


def test_run_stress_tests_no_overlap():
    """
    When the price series does not intersect the scenario window,
    period_covered should be False and metrics None.
    """
    dates = pd.date_range(start="2021-01-01", periods=10, freq="D")
    prices = pd.Series([100 + i for i in range(10)], index=dates)
    signals = pd.Series([0] * 10, index=dates)

    # Scenario completely outside the price series range
    scenario = StressScenario(
        name="no_overlap",
        label="No Overlap",
        start=date(2020, 5, 1),
        end=date(2020, 5, 10),
        description="Window before any price data",
    )

    results = run_stress_tests(
        signals=signals,
        prices=prices,
        scenarios=[scenario],
    )
    assert len(results) == 1
    res = results[0]
    assert res.period_covered is False
    assert res.metrics is None
    assert res.data_points == 0