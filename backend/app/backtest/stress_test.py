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


def _scenario_intersects_price_index(
    price_index: pd.Index,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> bool:
    """Return True if the scenario window overlaps any timestamps in the price index."""
    return ((price_index >= start_ts) & (price_index <= end_ts)).any()


def _prepare_slices(
    signals: pd.Series,
    prices: pd.Series,
    opens: pd.Series | None,
    volume: pd.Series | None,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None, pd.Series | None]:
    """Slice all input series to the scenario window."""
    s_signals = _slice_series(signals, start_ts, end_ts)
    s_prices = _slice_series(prices, start_ts, end_ts)
    s_opens = _slice_series(opens, start_ts, end_ts) if opens is not None else None
    s_volume = _slice_series(volume, start_ts, end_ts) if volume is not None else None
    return s_signals, s_prices, s_opens, s_volume


def _build_result(
    scenario: StressScenario,
    metrics: BacktestMetrics | None,
    period_covered: bool,
    data_points: int,
) -> StressResult:
    """Create a StressResult instance."""
    return StressResult(
        scenario=scenario,
        metrics=metrics,
        period_covered=period_covered,
        data_points=data_points,
    )


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

        # Fast check: skip if the window does not intersect the price series at all
        if not _scenario_intersects_price_index(price_index, start_ts, end_ts):
            results.append(_build_result(scenario, None, False, ZERO_DATA_POINTS))
            continue

        s_signals, s_prices, s_opens, s_volume = _prepare_slices(
            signals, prices, opens, volume, start_ts, end_ts
        )

        # Insufficient price data for this scenario
        if s_prices is None or len(s_prices) < MIN_DATA_POINTS:
            data_pts = len(s_prices) if s_prices is not None else ZERO_DATA_POINTS
            results.append(_build_result(scenario, None, False, data_pts))
            continue

        # Run backtest on the sliced data
        metrics = run_backtest(
            signals=s_signals,
            prices=s_prices,
            opens=s_opens,
            volume=s_volume,
            initial_equity=initial_equity,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )

        results.append(_build_result(scenario, metrics, True, len(s_prices)))

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