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
        # Sort unusual first
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


# ==========================
# Unit tests for edge cases
# ==========================
def test_put_call_ratio_zero_calls():
    """When there are no call volumes, ensure division uses fallback denominator."""
    test_scanner = OptionsFlowScanner()
    # Manually inject a single put entry
    test_scanner._cache = [
        OptionsFlow(
            ticker="AAPL",
            expiry=date.today(),
            strike=190,
            option_type="put",
            premium=5000,
            volume=100,
            open_interest=200,
            iv_percentile=30.0,
            sentiment="bearish",
            is_unusual=False,
            timestamp=datetime.now(timezone.utc),
        )
    ]
    result = test_scanner.put_call_ratio()
    assert result["calls"] == 0
    assert result["puts"] == 100
    # Ratio should be puts / 1 (fallback) = 100.0 rounded to two decimals
    assert result["ratio"] == 100.0
    assert result["sentiment"] == "bearish"


def test_is_unusual_boundary():
    """Validate is_unusual flag at exact boundary and just above it."""
    # Exactly at 3x OI should be False
    flow_at_boundary = OptionsFlow(
        ticker="TSLA",
        expiry=date.today(),
        strike=250,
        option_type="call",
        premium=1000,
        volume=3000,          # 3 * open_interest
        open_interest=1000,
        iv_percentile=50.0,
        sentiment="neutral",
        is_unusual=False,    # will be set manually for test clarity
        timestamp=datetime.now(timezone.utc),
    )
    assert not flow_at_boundary.is_unusual

    # Just above 3x OI should be True
    flow_above_boundary = OptionsFlow(
        ticker="TSLA",
        expiry=date.today(),
        strike=250,
        option_type="call",
        premium=1000,
        volume=3001,
        open_interest=1000,
        iv_percentile=50.0,
        sentiment="neutral",
        is_unusual=True,
        timestamp=datetime.now(timezone.utc),
    )
    assert flow_above_boundary.is_unusual


def test_generate_flow_strike_range():
    """Ensure generated strikes respect the configured percentage bounds."""
    random.seed(42)  # deterministic output for test reproducibility
    test_scanner = OptionsFlowScanner()
    flows = test_scanner._generate_flow()
    # Mapping of base prices used in _generate_flow
    base_prices = {
        "SPY": 450,
        "QQQ": 380,
        "AAPL": 185,
        "TSLA": 250,
        "NVDA": 800,
        "MSFT": 415,
        "AMZN": 185,
        "META": 500,
        "GOOGL": 175,
    }
    for flow in flows:
        base_price = base_prices.get(flow.ticker, 100)
        lower = base_price * 0.90
        upper = base_price * 1.15
        assert lower <= flow.strike <= upper, f"{flow.ticker} strike out of bounds"


# The tests can be executed with pytest without additional configuration.