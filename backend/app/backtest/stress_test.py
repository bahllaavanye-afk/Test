"""
Historical stress testing — overlay a strategy's signals on known crisis periods.

Tests how a strategy would have performed during the most severe market dislocations,
revealing tail-risk exposure that standard backtests can understate when they
average across calm and turbulent regimes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.backtest.engine import BacktestMetrics, run_backtest

# Configure module logger
logger = logging.getLogger(__name__)

if not logger.handlers:
    # Prevent duplicate handlers if root logger already configured elsewhere
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


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
    # None if the price data doesn't cover this period or an error occurred
    metrics: BacktestMetrics | None
    period_covered: bool
    data_points: int


def _slice_series(
    series: pd.Series | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | None:
    """Vectorized slice of a Series using .loc; returns None if input is None.

    If .loc raises an unexpected exception (e.g., due to a non‑datetime index),
    a boolean mask fallback is applied and the incident is logged.
    """
    if series is None:
        return None
    try:
        return series.loc[start:end]
    except (KeyError, TypeError, AttributeError) as e:
        logger.warning(
            "Series slicing with .loc failed for index type %s: %s. "
            "Falling back to boolean mask.",
            type(series.index).__name__,
            e,
        )
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
    # Input validation
    if not isinstance(signals, pd.Series):
        raise TypeError("signals must be a pandas Series")
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if opens is not None and not isinstance(opens, pd.Series):
        raise TypeError("opens must be a pandas Series or None")
    if volume is not None and not isinstance(volume, pd.Series):
        raise TypeError("volume must be a pandas Series or None")

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

        try:
            metrics = run_backtest(
                signals=s_signals,
                prices=s_prices,
                opens=s_opens,
                volume=s_volume,
                initial_equity=initial_equity,
                commission_pct=commission_pct,
                slippage_pct=slippage_pct,
            )
        except (ValueError, TypeError) as e:
            logger.error(
                "Backtest execution failed for scenario '%s': %s",
                scenario.name,
                e,
                exc_info=True,
            )
            metrics = None
            period_covered = False
        except Exception as e:
            logger.exception(
                "Unexpected error during backtest for scenario '%s'",
                scenario.name,
            )
            metrics = None
            period_covered = False
        else:
            period_covered = True

        results.append(
            StressResult(
                scenario=scenario,
                metrics=metrics,
                period_covered=period_covered,
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