"""Wheel strategy signal generator (cash-secured puts → covered calls)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field, validator


class WheelSignal(BaseModel):
    """Signal describing a cash‑secured put or covered call opportunity.

    Attributes
    ----------
    ticker: str
        Underlying ticker symbol.
    phase: Literal["sell_csp", "sell_cc"]
        Strategy phase – ``sell_csp`` for cash‑secured put, ``sell_cc`` for covered call.
    strike: float
        Option strike price (must be positive).
    expiry: date
        Expiration date of the option (must be today or later).
    premium: float
        Premium received per contract (in dollars, non‑negative).
    annualized_yield: float
        Annualized yield expressed as a percentage.
    iv_rank: float
        Implied volatility percentile rank (0‑100).
    delta: float
        Option delta; for puts: -0.3 → -0.1, for calls: 0.2 → 0.4.
    rationale: str
        Human‑readable explanation for why the signal was generated.
    """

    ticker: str = Field(..., description="Underlying ticker symbol.", example="SPY")
    phase: Literal["sell_csp", "sell_cc"] = Field(
        ...,
        description="Strategy phase – cash‑secured put or covered call.",
        example="sell_csp",
    )
    strike: float = Field(
        ...,
        gt=0,
        description="Option strike price (must be positive).",
        example=450.0,
    )
    expiry: date = Field(
        ...,
        description="Expiration date of the option (must be today or later).",
        example="2024-12-20",
    )
    premium: float = Field(
        ...,
        ge=0,
        description="Premium received per contract (in dollars, non‑negative).",
        example=12.5,
    )
    annualized_yield: float = Field(
        ...,
        ge=0,
        description="Annualized yield expressed as a percentage.",
        example=8.4,
    )
    iv_rank: float = Field(
        ...,
        ge=0,
        le=100,
        description="Implied volatility percentile rank (0‑100).",
        example=67.3,
    )
    delta: float = Field(
        ...,
        description="Option delta; for puts: -0.3 → -0.1, for calls: 0.2 → 0.4.",
        example=-0.25,
    )
    rationale: str = Field(
        ...,
        description="Human‑readable explanation for why the signal was generated.",
        example="IV rank 68% > 45, 35d to expiry, delta -0.25",
    )

    @validator("expiry")
    def expiry_not_past(cls, v: date) -> date:
        if v < date.today():
            raise ValueError("expiry date must be today or in the future")
        return v

    @validator("delta")
    def delta_in_range(cls, v: float, values) -> float:
        phase = values.get("phase")
        if phase == "sell_csp":
            if not -0.30 <= v <= -0.10:
                raise ValueError("delta for cash‑secured puts must be between -0.30 and -0.10")
        elif phase == "sell_cc":
            if not 0.20 <= v <= 0.40:
                raise ValueError("delta for covered calls must be between 0.20 and 0.40")
        return v

    def to_dict(self) -> dict:
        """Return a JSON‑serialisable dict with ISO‑formatted dates."""
        data = self.model_dump()
        data["expiry"] = self.expiry.isoformat()
        return data


def find_wheel_opportunities(tickers: list[str] | None = None) -> list[WheelSignal]:
    """
    Finds wheel strategy opportunities (high IV rank, 30-45 DTE, 0.25 delta).
    Production: use live options chain + IV percentile data.
    Demo: simulated realistic opportunities.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "NVDA", "AMD", "SPY"]

    signals: list[WheelSignal] = []
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