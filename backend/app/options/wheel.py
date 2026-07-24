"""Wheel strategy signal generator (cash‑secured puts → covered calls)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, List, Optional


@dataclass
class WheelSignal:
    """Container for a single wheel strategy signal."""
    ticker: str
    phase: Literal["sell_csp", "sell_cc"]  # cash‑secured put or covered call
    strike: float
    expiry: date
    premium: float          # premium received per contract (USD)
    annualized_yield: float # premium / (strike * 100) * (365 / DTE)  (%)
    iv_rank: float          # 0‑100; higher indicates richer premium
    delta: float            # option delta (negative for puts, positive for calls)
    rationale: str

    def to_dict(self) -> dict:
        """Serialise to a JSON‑compatible dict."""
        return {**self.__dict__, "expiry": self.expiry.isoformat()}

    def should_exit(
        self,
        underlying_price: float,
        days_elapsed: int,
        profit_target_pct: float = 0.5,
        loss_cutoff_pct: float = 0.2,
    ) -> bool:
        """
        Basic exit rule for wheel positions.

        - Exit if the underlying price has moved enough to capture the
          profit target on the short option.
        - Exit if the price moves against the position beyond the loss cutoff.
        - Exit automatically when the option is within 2 days of expiry.

        Parameters
        ----------
        underlying_price: float
            Current price of the underlying equity.
        days_elapsed: int
            Number of days passed since the position was opened.
        profit_target_pct: float
            Desired profit as a fraction of the premium (default 50%).
        loss_cutoff_pct: float
            Maximum acceptable loss as a fraction of the premium (default 20%).

        Returns
        -------
        bool
            True if the position should be closed, False otherwise.
        """
        # Time‑based exit
        if (self.expiry - date.today()).days <= 2:
            return True

        # Profit / loss calculation based on premium received
        pnl = (self.premium - self._current_option_price(underlying_price)) / self.premium

        if pnl >= profit_target_pct:
            return True
        if pnl <= -loss_cutoff_pct:
            return True

        return False

    def _current_option_price(self, underlying_price: float) -> float:
        """
        Rough approximation of the current option price using intrinsic value.
        For puts, intrinsic = max(strike - underlying, 0); for calls,
        intrinsic = max(underlying - strike, 0). We add a small time value
        component proportional to remaining DTE.

        This is a lightweight proxy used for exit decisions; the production
        system will replace it with a proper pricing model.
        """
        dte = max((self.expiry - date.today()).days, 0)
        if self.phase == "sell_csp":  # cash‑secured put
            intrinsic = max(self.strike - underlying_price, 0)
        else:  # covered call
            intrinsic = max(underlying_price - self.strike, 0)

        time_value = self.premium * (dte / max((self.expiry - date.today()).days + 1, 1)) * 0.1
        return intrinsic + time_value


def _price_trend_confirmation(ticker: str) -> bool:
    """
    Simple confirmation filter based on recent price momentum.
    In a real environment this would query recent price data; here we
    simulate a modest bias toward continuation.

    Returns True if the recent trend supports the short option direction.
    """
    # Simulated recent 5‑day price change (%)
    recent_change = random.uniform(-3.0, 3.0)
    # For cash‑secured puts we prefer a non‑negative trend;
    # for covered calls we prefer a non‑negative trend as well.
    return recent_change >= -1.0  # allow slight pull‑back


def find_wheel_opportunities(tickers: Optional[List[str]] = None) -> List[WheelSignal]:
    """
    Generate wheel strategy opportunities with tighter entry criteria
    and a confirmation filter.

    Entry criteria
    --------------
    * IV rank > 60 (rich premium environment)
    * DTE between 30 and 45 days
    * Delta within tighter bands:
        - puts: -0.25 to -0.15
        - calls: 0.20 to 0.35
    * Annualized yield > 5%
    * Recent price trend confirmation

    Returns
    -------
    List[WheelSignal]
        Sorted by descending annualized yield, limited to top 10 signals.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "NVDA", "AMD", "SPY", "TSLA", "META", "AMZN"]

    today = date.today()
    # Base price dictionary used for demo purposes only.
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

    signals: List[WheelSignal] = []

    for ticker in tickers:
        price = base_prices.get(ticker, 100)

        # IV rank – require > 60 for quality premium
        iv_rank = random.uniform(40, 90)
        if iv_rank <= 60:
            continue

        # DTE selection within 30‑45 day window
        dte = random.choice([30, 35, 40, 45])
        expiry = today + timedelta(days=dte)

        # Delta tighter range for puts
        delta = round(random.uniform(-0.25, -0.15), 2)

        # Strike placed ~5‑10% OTM for puts (adjusted by delta)
        strike = round(price * (1 + delta * 0.4), 0)
        if strike <= 0:
            continue  # safeguard against nonsensical strikes

        # Premium per share – realistic range based on price
        premium_per_share = round(price * random.uniform(0.008, 0.025), 2)
        if premium_per_share <= 0:
            continue

        # Annualized yield calculation (%)
        ann_yield = round(premium_per_share / strike * 365 / dte * 100, 1)

        # Enforce a minimum yield threshold
        if ann_yield < 5.0:
            continue

        # Confirmation filter based on recent price trend
        if not _price_trend_confirmation(ticker):
            continue

        rationale = (
            f"IV rank {iv_rank:.0f}% > 60, {dte}d DTE, delta {delta}, "
            f"annualized yield {ann_yield:.1f}%"
        )

        signals.append(
            WheelSignal(
                ticker=ticker,
                phase="sell_csp",
                strike=strike,
                expiry=expiry,
                premium=round(premium_per_share * 100, 2),  # per contract (100 shares)
                annualized_yield=ann_yield,
                iv_rank=round(iv_rank, 1),
                delta=delta,
                rationale=rationale,
            )
        )

    # Sort by most attractive yield and cap the list
    signals.sort(key=lambda s: -s.annualized_yield)
    return signals[:10]