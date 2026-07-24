"""Wheel strategy signal generator (cash-secured puts → covered calls)."""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class WheelSignal:
    ticker: str
    phase: Literal["sell_csp", "sell_cc"]  # cash-secured put or covered call
    strike: float
    expiry: date
    premium: float          # premium received per contract
    annualized_yield: float # premium / (strike * 100) * (365 / dte)
    iv_rank: float          # 0-100; want > 50 for good premiums
    delta: float            # -0.3 to -0.1 for puts, 0.2 to 0.4 for calls
    rationale: str

    def to_dict(self) -> dict:
        return {**self.__dict__, "expiry": self.expiry.isoformat()}


def find_wheel_opportunities(tickers: list[str] | None = None) -> list[WheelSignal]:
    """
    Finds wheel strategy opportunities (high IV rank, 30-45 DTE, 0.25 delta).
    Production: use live options chain + IV percentile data.
    Demo: simulated realistic opportunities.
    """
    start_time = time.perf_counter()

    if tickers is None:
        tickers = ["AAPL", "MSFT", "NVDA", "AMD", "SPY"]

    signals = []
    today = date.today()
    base_prices = {
        "AAPL": 185,
        "MSFT": 415,
        "NVDA": 800,
        "AMD": 170,
        "SPY": 450,
        "TSLA": 250,
        "META": 500,
        "AMZN": 185,
    }

    for ticker in tickers:
        price = base_prices.get(ticker, 100)
        iv_rank = random.uniform(40, 90)
        if iv_rank < 45:
            continue  # skip low IV rank — bad premium
        dte = random.choice([21, 28, 35, 42])
        expiry = today + timedelta(days=dte)
        delta = round(random.uniform(-0.30, -0.20), 2)
        strike = round(price * (1 + delta * 0.5), 0)  # ~10-15% OTM
        premium_per_share = round(price * random.uniform(0.008, 0.025), 2)
        ann_yield = round(premium_per_share / strike * 365 / dte * 100, 1)
        signals.append(
            WheelSignal(
                ticker=ticker,
                phase="sell_csp",
                strike=strike,
                expiry=expiry,
                premium=round(premium_per_share * 100, 2),
                annualized_yield=ann_yield,
                iv_rank=round(iv_rank, 1),
                delta=delta,
                rationale=(
                    f"IV rank {iv_rank:.0f}% > 45, {dte}d to expiry, delta {delta}"
                ),
            )
        )

    signals.sort(key=lambda s: -s.annualized_yield)
    top_signals = signals[:10]

    # Monitoring metrics
    execution_time_ms = (time.perf_counter() - start_time) * 1000
    total_premium = sum(s.premium for s in top_signals)
    logger.info(
        "wheel_opportunities_generated",
        extra={
            "signal_count": len(top_signals),
            "execution_time_ms": round(execution_time_ms, 2),
            "total_premium": round(total_premium, 2),
        },
    )

    return top_signals