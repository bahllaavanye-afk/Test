"""Wheel strategy signal generator (cash-secured puts → covered calls)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal
import random


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


def _get_base_price(ticker: str) -> float:
    """Return a default underlying price for the given ticker."""
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
    return base_prices.get(ticker, 100)


def _generate_iv_rank() -> float:
    """Generate a random IV rank between 40 and 90 (inclusive)."""
    return random.uniform(40, 90)


def _choose_dte() -> int:
    """Select a random days‑to‑expiry from the typical wheel range."""
    return random.choice([21, 28, 35, 42])


def _calculate_delta() -> float:
    """Generate a random delta for a cash‑secured put (negative value)."""
    return round(random.uniform(-0.30, -0.20), 2)


def _calculate_strike(price: float, delta: float) -> float:
    """
    Approximate an out‑of‑the‑money strike.
    The factor 0.5 mirrors the original heuristic.
    """
    return round(price * (1 + delta * 0.5), 0)


def _calculate_premium_per_share(price: float) -> float:
    """Generate a realistic premium per share based on price."""
    return round(price * random.uniform(0.008, 0.025), 2)


def _calculate_annualized_yield(premium_per_share: float, strike: float, dte: int) -> float:
    """Compute annualized yield as a percentage."""
    return round(premium_per_share / strike * 365 / dte * 100, 1)


def _build_signal(
    ticker: str,
    price: float,
    iv_rank: float,
    dte: int,
    expiry: date,
    delta: float,
    strike: float,
    premium_per_share: float,
    ann_yield: float,
) -> WheelSignal:
    """Create a WheelSignal instance with a formatted rationale."""
    rationale = (
        f"IV rank {iv_rank:.0f}% > 45, {dte}d to expiry, delta {delta}"
    )
    return WheelSignal(
        ticker=ticker,
        phase="sell_csp",
        strike=strike,
        expiry=expiry,
        premium=round(premium_per_share * 100, 2),
        annualized_yield=ann_yield,
        iv_rank=round(iv_rank, 1),
        delta=delta,
        rationale=rationale,
    )


def find_wheel_opportunities(tickers: list[str] | None = None) -> list[WheelSignal]:
    """
    Finds wheel strategy opportunities (high IV rank, 30‑45 DTE, 0.25 delta).
    Production: use live options chain + IV percentile data.
    Demo: simulated realistic opportunities.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "NVDA", "AMD", "SPY"]

    today = date.today()
    signals: list[WheelSignal] = []

    for ticker in tickers:
        price = _get_base_price(ticker)
        iv_rank = _generate_iv_rank()
        if iv_rank < 45:
            continue  # skip low IV rank — bad premium

        dte = _choose_dte()
        expiry = today + timedelta(days=dte)

        delta = _calculate_delta()
        strike = _calculate_strike(price, delta)
        premium_per_share = _calculate_premium_per_share(price)
        ann_yield = _calculate_annualized_yield(premium_per_share, strike, dte)

        signals.append(
            _build_signal(
                ticker=ticker,
                price=price,
                iv_rank=iv_rank,
                dte=dte,
                expiry=expiry,
                delta=delta,
                strike=strike,
                premium_per_share=premium_per_share,
                ann_yield=ann_yield,
            )
        )

    signals.sort(key=lambda s: -s.annualized_yield)
    return signals[:10]