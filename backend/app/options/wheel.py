"""Wheel strategy signal generator (cash-secured puts → covered calls).

This module provides functionality to generate synthetic wheel strategy signals for a
set of tickers. The signals are intended for demo or testing environments and are
based on randomised but realistic‑looking option parameters such as implied‑volatility
rank, days‑to‑expiry (DTE) and delta. The generated signals are sorted by annualized
yield and limited to the top ten candidates.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, List


@dataclass
class WheelSignal:
    """Container for a single wheel strategy signal.

    Attributes
    ----------
    ticker: str
        Underlying equity ticker.
    phase: Literal["sell_csp", "sell_cc"]
        The wheel phase – either selling a cash‑secured put (``sell_csp``) or a
        covered call (``sell_cc``).
    strike: float
        Option strike price.
    expiry: date
        Expiration date of the option contract.
    premium: float
        Premium received per contract (in dollars, i.e. 100 × per‑share premium).
    annualized_yield: float
        Annualized yield expressed as a percentage; calculated as
        ``premium / (strike * 100) * (365 / DTE)``.
    iv_rank: float
        Implied‑volatility percentile rank (0‑100). Values > 50 are generally
        considered attractive for premium generation.
    delta: float
        Option delta. For puts this is a negative value (≈ -0.3 to -0.1);
        for calls a positive value (≈ 0.2 to 0.4).
    rationale: str
        Human‑readable explanation of why the signal was generated.
    """

    ticker: str
    phase: Literal["sell_csp", "sell_cc"]
    strike: float
    expiry: date
    premium: float          # premium received per contract
    annualized_yield: float # premium / (strike * 100) * (365 / dte)
    iv_rank: float          # 0-100; want > 50 for good premiums
    delta: float            # -0.3 to -0.1 for puts, 0.2 to 0.4 for calls
    rationale: str

    def to_dict(self) -> dict:
        """Return a JSON‑serialisable representation of the signal.

        The ``expiry`` date is converted to ISO‑8601 format to ensure compatibility
        with downstream services that expect primitive types.

        Returns
        -------
        dict
            Dictionary containing all dataclass fields with ``expiry`` as a string.
        """
        return {**self.__dict__, "expiry": self.expiry.isoformat()}


def find_wheel_opportunities(tickers: list[str] | None = None) -> List[WheelSignal]:
    """Generate wheel strategy opportunities for a collection of tickers.

    The function simulates realistic option chain data for each ticker, filters
    out low‑IV‑rank contracts, and creates a :class:`WheelSignal` instance for the
    most attractive cash‑secured put opportunity (the covered‑call phase is not
    generated in this demo).

    Parameters
    ----------
    tickers : list[str] | None, optional
        List of ticker symbols to evaluate. If ``None``, a default list of
        frequently traded equities is used.

    Returns
    -------
    List[WheelSignal]
        Up to ten wheel signals sorted by descending annualized yield.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "NVDA", "AMD", "SPY"]

    signals: List[WheelSignal] = []
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
                rationale=f"IV rank {iv_rank:.0f}% > 45, {dte}d to expiry, delta {delta}",
            )
        )

    signals.sort(key=lambda s: -s.annualized_yield)
    return signals[:10]