"""Options flow scanner — unusual activity detection.

Provides a simulated data source for options flow, useful for demo or paper‑trading
environments. In production the scanner would be replaced with a live feed from
providers such as Tradier, Polygon, or CBOE.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timezone, date, timedelta
from typing import Literal, List, Dict, Any


@dataclass
class OptionsFlow:
    """Container for a single options flow record.

    Attributes
    ----------
    ticker: str
        Underlying equity ticker symbol.
    expiry: date
        Expiration date of the option contract.
    strike: float
        Strike price of the option.
    option_type: Literal["call", "put"]
        Type of the option – either ``"call"`` or ``"put"``.
    premium: float
        Total premium paid (USD) for the trade.
    volume: int
        Number of contracts traded.
    open_interest: int
        Current open interest for the contract.
    iv_percentile: float
        Implied volatility percentile (0‑100).
    sentiment: Literal["bullish", "bearish", "neutral"]
        Directional sentiment inferred from the trade.
    is_unusual: bool
        Flag indicating unusually high volume (volume > 3× average OI).
    timestamp: datetime
        UTC timestamp when the flow was generated.
    """

    ticker: str
    expiry: date
    strike: float
    option_type: Literal["call", "put"]
    premium: float          # total premium in USD
    volume: int
    open_interest: int
    iv_percentile: float    # 0-100
    sentiment: Literal["bullish", "bearish", "neutral"]
    is_unusual: bool        # volume > 3x avg OI
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON‑serialisable representation of the flow."""
        return {
            "ticker": self.ticker,
            "expiry": self.expiry.isoformat(),
            "strike": self.strike,
            "option_type": self.option_type,
            "premium": self.premium,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "iv_percentile": self.iv_percentile,
            "sentiment": self.sentiment,
            "is_unusual": self.is_unusual,
            "timestamp": self.timestamp.isoformat(),
        }


class OptionsFlowScanner:
    """
    Scans for unusual options activity.

    In a production environment this class would connect to a live data feed.
    For demo or paper‑trading use it generates realistic simulated flow data.
    """

    WATCHLIST: List[str] = [
        "AAPL", "TSLA", "SPY", "QQQ", "NVDA",
        "MSFT", "AMZN", "META", "GOOGL",
    ]

    def __init__(self) -> None:
        """Initialize internal cache and timestamp."""
        self._cache: List[OptionsFlow] = []
        self._last_refresh: datetime = datetime.min.replace(tzinfo=timezone.utc)

    async def scan(self, refresh_seconds: int = 60) -> List[OptionsFlow]:
        """
        Retrieve the latest options flow.

        Parameters
        ----------
        refresh_seconds: int, optional
            Minimum number of seconds before a fresh generation is performed.
            Defaults to 60.

        Returns
        -------
        List[OptionsFlow]
            A list of up to 50 flow records, sorted with unusual activity first.
        """
        now = datetime.now(timezone.utc)
        if (now - self._last_refresh).total_seconds() < refresh_seconds and self._cache:
            return self._cache
        self._cache = self._generate_flow()
        self._last_refresh = now
        return self._cache

    def _generate_flow(self) -> List[OptionsFlow]:
        """Generate simulated options flow for demo (replace with live data feed)."""
        flows: List[OptionsFlow] = []
        today = date.today()
        for ticker in self.WATCHLIST:
            for _ in range(random.randint(3, 8)):
                days_out = random.choice([7, 14, 21, 30, 45, 60])
                expiry = today + timedelta(days=days_out)
                base_price = {
                    "SPY": 450, "QQQ": 380, "AAPL": 185, "TSLA": 250,
                    "NVDA": 800, "MSFT": 415, "AMZN": 185, "META": 500,
                    "GOOGL": 175,
                }.get(ticker, 100)
                strike_pct = random.uniform(0.90, 1.15)
                strike = round(base_price * strike_pct, 0)
                oi = random.randint(1000, 50000)
                vol = int(oi * random.uniform(0.1, 5.0))
                is_unusual = vol > oi * 3
                opt_type = random.choice(["call", "put"])
                premium = round(random.uniform(0.5, 50) * vol * 100, 0)
                iv_pct = random.uniform(20, 95)
                if opt_type == "call" and strike_pct < 1.05:
                    sentiment = "bullish"
                elif opt_type == "put" and strike_pct > 0.95:
                    sentiment = "bearish"
                else:
                    sentiment = "neutral"
                flows.append(
                    OptionsFlow(
                        ticker=ticker,
                        expiry=expiry,
                        strike=strike,
                        option_type=opt_type,
                        premium=premium,
                        volume=vol,
                        open_interest=oi,
                        iv_percentile=round(iv_pct, 1),
                        sentiment=sentiment,
                        is_unusual=is_unusual,
                        timestamp=datetime.now(timezone.utc),
                    )
                )
        # Sort unusual first, then by premium descending
        flows.sort(key=lambda f: (not f.is_unusual, -f.premium))
        return flows[:50]

    def put_call_ratio(self) -> Dict[str, Any]:
        """
        Compute the aggregate put/call volume ratio for the cached flow.

        Returns
        -------
        dict
            Contains ``ratio`` (float), ``calls`` (int), ``puts`` (int) and a
            derived ``sentiment`` string based on the ratio.
        """
        if not self._cache:
            return {"ratio": 0.0, "calls": 0, "puts": 0}
        calls = sum(f.volume for f in self._cache if f.option_type == "call")
        puts = sum(f.volume for f in self._cache if f.option_type == "put")
        ratio = round(puts / max(calls, 1), 2)
        sentiment = (
            "bearish"
            if ratio > 1.2
            else "bullish"
            if ratio < 0.8
            else "neutral"
        )
        return {"ratio": ratio, "calls": calls, "puts": puts, "sentiment": sentiment}


scanner = OptionsFlowScanner()