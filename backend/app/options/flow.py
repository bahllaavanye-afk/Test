"""Options flow scanner — unusual activity detection.

Provides a simulated options flow generator for demo environments and a simple
interface to retrieve the latest flow data. In production the scanner would be
replaced with a live data feed from a broker or market data provider.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Literal


@dataclass
class OptionsFlow:
    """Data model representing a single options flow event.

    Attributes
    ----------
    ticker: str
        Underlying ticker symbol.
    expiry: date
        Expiration date of the option contract.
    strike: float
        Strike price of the option.
    option_type: Literal["call", "put"]
        Type of the option – either ``"call"`` or ``"put"``.
    premium: float
        Total premium paid (USD) for the flow.
    volume: int
        Number of contracts traded.
    open_interest: int
        Current open interest for the option.
    iv_percentile: float
        Implied volatility percentile (0‑100) relative to historical data.
    sentiment: Literal["bullish", "bearish", "neutral"]
        Inferred market sentiment based on the flow.
    is_unusual: bool
        Flag indicating whether the flow is considered unusual (volume > 3× average OI).
    timestamp: datetime
        Timestamp of when the flow was generated.
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

    def to_dict(self) -> dict[str, object]:
        """Convert the dataclass to a plain dictionary suitable for JSON serialisation.

        Returns
        -------
        dict
            Dictionary with ISO‑formatted dates and timestamps.
        """
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

    In production this class would connect to a live data feed (e.g. Tradier,
    Polygon, CBOE). For demo or paper‑trading environments it generates realistic
    simulated flow data.
    """

    WATCHLIST = [
        "AAPL",
        "TSLA",
        "SPY",
        "QQQ",
        "NVDA",
        "MSFT",
        "AMZN",
        "META",
        "GOOGL",
    ]

    def __init__(self) -> None:
        """Initialize the scanner with an empty cache."""
        self._cache: list[OptionsFlow] = []
        self._last_refresh = datetime.min.replace(tzinfo=timezone.utc)

    async def scan(self, refresh_seconds: int = 60) -> list[OptionsFlow]:
        """Return the latest options flow, refreshing the cache if needed.

        Parameters
        ----------
        refresh_seconds: int, optional
            Minimum number of seconds before the cached data is considered stale.
            Defaults to ``60``.

        Returns
        -------
        list[OptionsFlow]
            List of the most recent flow events (up to 50 entries).
        """
        now = datetime.now(timezone.utc)
        if (now - self._last_refresh).total_seconds() < refresh_seconds and self._cache:
            return self._cache
        self._cache = self._generate_flow()
        self._last_refresh = now
        return self._cache

    def _generate_flow(self) -> list[OptionsFlow]:
        """Generate simulated options flow for demo (replace with live data feed).

        Returns
        -------
        list[OptionsFlow]
            A list of generated flow events sorted with unusual activity first.
        """
        flows: list[OptionsFlow] = []
        today = date.today()
        for ticker in self.WATCHLIST:
            for _ in range(random.randint(3, 8)):
                days_out = random.choice([7, 14, 21, 30, 45, 60])
                expiry = today + timedelta(days=days_out)
                base_price = {
                    "SPY": 450,
                    "QQQ": 380,
                    "AAPL": 185,
                    "TSLA": 250,
                    "NVDA": 800,
                    "MSFT": 415,
                    "AMZN": 185,
                    "META": 500,
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

    def put_call_ratio(self) -> dict[str, object]:
        """Calculate the put‑call volume ratio for the cached flow data.

        Returns
        -------
        dict
            Mapping containing the ratio, raw call/put volumes, and an inferred
            sentiment label.
        """
        if not self._cache:
            return {"ratio": 0.0, "calls": 0, "puts": 0, "sentiment": "neutral"}
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