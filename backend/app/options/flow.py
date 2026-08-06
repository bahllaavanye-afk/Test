"""Options flow scanner — unusual activity detection."""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
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
    In production: connect to Tradier, Polygon, or CBOE data feed.
    For demo/paper trading: generates realistic simulated flow data.
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
        self._cache: list[OptionsFlow] = []
        self._last_refresh = datetime.min.replace(tzinfo=timezone.utc)

    async def scan(self, refresh_seconds: int = 60) -> list[OptionsFlow]:
        now = datetime.now(timezone.utc)
        if (now - self._last_refresh).total_seconds() < refresh_seconds and self._cache:
            return self._cache
        self._cache = self._generate_flow()
        self._last_refresh = now
        return self._cache

    # --------------------------------------------------------------------- #
    # Helper methods for flow generation
    # --------------------------------------------------------------------- #

    @staticmethod
    def _base_price(ticker: str) -> float:
        """Return a representative base price for a ticker."""
        return {
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

    @staticmethod
    def _select_days_out() -> int:
        """Select a random number of days until expiry."""
        return random.choice([7, 14, 21, 30, 45, 60])

    @staticmethod
    def _strike_price(base_price: float) -> float:
        """Calculate a strike price as a percentage of the base price."""
        strike_pct = random.uniform(0.90, 1.15)
        return round(base_price * strike_pct, 0), strike_pct

    @staticmethod
    def _open_interest() -> int:
        """Generate a random open interest."""
        return random.randint(1000, 50000)

    @staticmethod
    def _volume(oi: int) -> int:
        """Generate volume based on open interest."""
        return int(oi * random.uniform(0.1, 5.0))

    @staticmethod
    def _option_type() -> Literal["call", "put"]:
        """Randomly choose option type."""
        return random.choice(["call", "put"])

    @staticmethod
    def _premium(volume: int) -> float:
        """Calculate premium based on volume."""
        return round(random.uniform(0.5, 50) * volume * 100, 0)

    @staticmethod
    def _iv_percentile() -> float:
        """Generate an implied volatility percentile."""
        return round(random.uniform(20, 95), 1)

    @staticmethod
    def _determine_sentiment(opt_type: Literal["call", "put"], strike_pct: float) -> Literal["bullish", "bearish", "neutral"]:
        """Derive sentiment from option type and strike proximity."""
        if opt_type == "call" and strike_pct < 1.05:
            return "bullish"
        if opt_type == "put" and strike_pct > 0.95:
            return "bearish"
        return "neutral"

    def _create_flow(self, ticker: str, today: date) -> OptionsFlow:
        """Create a single OptionsFlow instance for a given ticker."""
        days_out = self._select_days_out()
        expiry = today + timedelta(days=days_out)

        base_price = self._base_price(ticker)
        strike, strike_pct = self._strike_price(base_price)

        oi = self._open_interest()
        vol = self._volume(oi)
        is_unusual = vol > oi * 3

        opt_type = self._option_type()
        premium = self._premium(vol)
        iv_pct = self._iv_percentile()
        sentiment = self._determine_sentiment(opt_type, strike_pct)

        return OptionsFlow(
            ticker=ticker,
            expiry=expiry,
            strike=strike,
            option_type=opt_type,
            premium=premium,
            volume=vol,
            open_interest=oi,
            iv_percentile=iv_pct,
            sentiment=sentiment,
            is_unusual=is_unusual,
            timestamp=datetime.now(timezone.utc),
        )

    def _generate_flow(self) -> list[OptionsFlow]:
        """Generate simulated options flow for demo (replace with live data feed)."""
        flows: list[OptionsFlow] = []
        today = date.today()
        for ticker in self.WATCHLIST:
            for _ in range(random.randint(3, 8)):
                flows.append(self._create_flow(ticker, today))

        # Sort unusual first, then by descending premium
        flows.sort(key=lambda f: (not f.is_unusual, -f.premium))
        return flows[:50]

    def put_call_ratio(self) -> dict:
        if not self._cache:
            return {"ratio": 0.0, "calls": 0, "puts": 0}
        calls = sum(f.volume for f in self._cache if f.option_type == "call")
        puts = sum(f.volume for f in self._cache if f.option_type == "put")
        ratio = round(puts / max(calls, 1), 2)
        return {
            "ratio": ratio,
            "calls": calls,
            "puts": puts,
            "sentiment": "bearish" if ratio > 1.2 else "bullish" if ratio < 0.8 else "neutral",
        }


scanner = OptionsFlowScanner()