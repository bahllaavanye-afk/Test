"""
Profit forecasting math — pure functions, no DB/network so they unit-test cleanly.

Given a realized daily-PnL series, project forward over weekly / monthly / yearly
horizons with confidence bands. The projection treats daily PnL as i.i.d. draws:
expected PnL scales linearly with the horizon (mean * h) while uncertainty scales
with its square root (std * sqrt(h)) — the standard random-walk variance scaling.
These are statistical projections from observed data, explicitly labelled as such;
nothing here is hardcoded or guaranteed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

# Trading-day horizons.
TRADING_DAYS = {"weekly": 5, "monthly": 21, "yearly": 252}
ANNUALISATION = math.sqrt(252)


@dataclass
class PnLStats:
    n_days: int
    total_pnl: float
    mean_daily: float
    std_daily: float
    sharpe_annual: float
    positive_day_rate: float


@dataclass
class Projection:
    horizon: str
    horizon_days: int
    expected_pnl: float
    low_pnl: float          # expected - z*sigma_h
    high_pnl: float         # expected + z*sigma_h
    confidence_z: float


def summarize_daily_pnl(daily_pnls: list[float]) -> PnLStats:
    n = len(daily_pnls)
    if n == 0:
        return PnLStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total = float(sum(daily_pnls))
    mean = total / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in daily_pnls) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    sharpe = (mean / std * ANNUALISATION) if std > 1e-12 else 0.0
    pos_rate = sum(1 for x in daily_pnls if x > 0) / n
    return PnLStats(
        n_days=n, total_pnl=round(total, 2), mean_daily=round(mean, 4),
        std_daily=round(std, 4), sharpe_annual=round(sharpe, 3),
        positive_day_rate=round(pos_rate, 3),
    )


def project(daily_pnls: list[float], horizon_days: int, horizon: str = "", z: float = 1.0) -> Projection:
    """Project cumulative PnL over `horizon_days` with a ±z·sigma band."""
    stats = summarize_daily_pnl(daily_pnls)
    expected = stats.mean_daily * horizon_days
    sigma_h = stats.std_daily * math.sqrt(horizon_days)
    return Projection(
        horizon=horizon or f"{horizon_days}d",
        horizon_days=horizon_days,
        expected_pnl=round(expected, 2),
        low_pnl=round(expected - z * sigma_h, 2),
        high_pnl=round(expected + z * sigma_h, 2),
        confidence_z=z,
    )


def build_forecast(
    daily_pnls: list[float],
    by_desk: dict[str, list[float]] | None = None,
    z: float = 1.0,
) -> dict:
    """
    Full forecast: overall stats + weekly/monthly/yearly projections, plus a
    per-desk projection breakdown when desk-level daily series are supplied.

    Returns a dict ready to serialize to an API response or a Slack summary.
    Flags `sufficient_data` False when there are too few days to be meaningful.
    """
    stats = summarize_daily_pnl(daily_pnls)
    projections = {
        name: asdict(project(daily_pnls, days, horizon=name, z=z))
        for name, days in TRADING_DAYS.items()
    }

    desk_block: dict[str, dict] = {}
    if by_desk:
        for desk, series in by_desk.items():
            desk_stats = summarize_daily_pnl(series)
            desk_block[desk] = {
                "stats": asdict(desk_stats),
                "yearly_expected_pnl": asdict(project(series, TRADING_DAYS["yearly"],
                                                      horizon="yearly", z=z))["expected_pnl"],
            }

    return {
        "sufficient_data": stats.n_days >= 5,
        "stats": asdict(stats),
        "projections": projections,
        "by_desk": desk_block,
        "method": "i.i.d. daily-PnL projection (mean·h ± z·std·√h); statistical estimate, not a guarantee",
    }
