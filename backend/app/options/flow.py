"""Options flow scanner — unusual activity detection."""
from __future__ import annotations
import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timezone, date, timedelta
from typing import Literal


@dataclass
class OptionsFlow:
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

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker, "expiry": self.expiry.isoformat(),
            "strike": self.strike, "option_type": self.option_type,
            "premium": self.premium, "volume": self.volume,
            "open_interest": self.open_interest, "iv_percentile": self.iv_percentile,
            "sentiment": self.sentiment, "is_unusual": self.is_unusual,
            "timestamp": self.timestamp.isoformat(),
        }


class OptionsFlowScanner:
    """
    Scans for unusual options activity.
    In production: connect to Tradier, Polygon, or CBOE data feed.
    For demo/paper trading: generates realistic simulated flow data.
    """

    WATCHLIST = ["AAPL", "TSLA", "SPY", "QQQ", "NVDA", "MSFT", "AMZN", "META", "GOOGL"]

    def __init__(self):
        self._cache: list[OptionsFlow] = []
        self._last_refresh = datetime.min.replace(tzinfo=timezone.utc)

    async def scan(self, refresh_seconds: int = 60) -> list[OptionsFlow]:
        now = datetime.now(timezone.utc)
        if (now - self._last_refresh).total_seconds() < refresh_seconds and self._cache:
            return self._cache
        self._cache = self._generate_flow()
        self._last_refresh = now
        return self._cache

    def _generate_flow(self) -> list[OptionsFlow]:
        """Generate simulated options flow for demo (replace with live data feed)."""
        flows = []
        today = date.today()
        for ticker in self.WATCHLIST:
            for _ in range(random.randint(3, 8)):
                days_out = random.choice([7, 14, 21, 30, 45, 60])
                expiry = today + timedelta(days=days_out)
                base_price = {"SPY": 450, "QQQ": 380, "AAPL": 185, "TSLA": 250,
                              "NVDA": 800, "MSFT": 415, "AMZN": 185, "META": 500,
                              "GOOGL": 175}.get(ticker, 100)
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
                flows.append(OptionsFlow(
                    ticker=ticker, expiry=expiry, strike=strike, option_type=opt_type,
                    premium=premium, volume=vol, open_interest=oi,
                    iv_percentile=round(iv_pct, 1), sentiment=sentiment,
                    is_unusual=is_unusual, timestamp=datetime.now(timezone.utc),
                ))
        # Sort unusual first
        flows.sort(key=lambda f: (not f.is_unusual, -f.premium))
        return flows[:50]

    def put_call_ratio(self) -> dict:
        if not self._cache:
            return {"ratio": 0.0, "calls": 0, "puts": 0}
        calls = sum(f.volume for f in self._cache if f.option_type == "call")
        puts = sum(f.volume for f in self._cache if f.option_type == "put")
        ratio = round(puts / max(calls, 1), 2)
        return {"ratio": ratio, "calls": calls, "puts": puts, "sentiment": "bearish" if ratio > 1.2 else "bullish" if ratio < 0.8 else "neutral"}


scanner = OptionsFlowScanner()
