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


# ==================== Unit Tests ====================

import unittest
from unittest.mock import patch


class TestOptionsFlowScanner(unittest.TestCase):
    def setUp(self):
        self.scanner = OptionsFlowScanner()

    def test_put_call_ratio_empty_cache(self):
        """When cache is empty, ratio and counts should be zero."""
        result = self.scanner.put_call_ratio()
        self.assertEqual(result["ratio"], 0.0)
        self.assertEqual(result["calls"], 0)
        self.assertEqual(result["puts"], 0)

    def test_put_call_ratio_calls_zero(self):
        """When there are puts but no calls, division should use max(calls,1)."""
        # Create a flow with only puts
        flow = OptionsFlow(
            ticker="AAPL",
            expiry=date.today(),
            strike=150,
            option_type="put",
            premium=1000,
            volume=2000,
            open_interest=500,
            iv_percentile=30.0,
            sentiment="bearish",
            is_unusual=False,
            timestamp=datetime.now(timezone.utc),
        )
        self.scanner._cache = [flow]
        result = self.scanner.put_call_ratio()
        self.assertEqual(result["calls"], 0)
        self.assertEqual(result["puts"], 2000)
        self.assertEqual(result["ratio"], round(2000 / 1, 2))

    def test_is_unusual_boundary(self):
        """Volume exactly three times open_interest should NOT be marked unusual."""
        oi = 1000
        vol = oi * 3  # boundary condition
        flow = OptionsFlow(
            ticker="TSLA",
            expiry=date.today(),
            strike=300,
            option_type="call",
            premium=5000,
            volume=vol,
            open_interest=oi,
            iv_percentile=45.0,
            sentiment="neutral",
            is_unusual=vol > oi * 3,  # should be False
            timestamp=datetime.now(timezone.utc),
        )
        self.assertFalse(flow.is_unusual)

    @patch("random.randint", return_value=3)
    @patch("random.choice", side_effect=lambda x: x[0])  # deterministic days_out
    @patch("random.uniform", side_effect=[0.90, 0.5, 20])  # strike_pct, volume factor, iv_pct
    @patch("random.choice", side_effect=lambda x: "call")  # option_type
    def test_generate_flow_deterministic(self, mock_randint, mock_choice_days, mock_uniform, mock_option_type):
        """Generate flow with mocked randomness to ensure deterministic output."""
        flows = self.scanner._generate_flow()
        # Verify that at least one flow is generated and fields are within expected ranges
        self.assertTrue(len(flows) > 0)
        for f in flows:
            self.assertIn(f.ticker, self.scanner.WATCHLIST)
            self.assertIsInstance(f.expiry, date)
            self.assertGreaterEqual(f.strike, 0)
            self.assertIn(f.option_type, ("call", "put"))
            self.assertGreaterEqual(f.premium, 0)
            self.assertGreaterEqual(f.volume, 0)
            self.assertGreaterEqual(f.iv_percentile, 0)
            self.assertIn(f.sentiment, ("bullish", "bearish", "neutral"))
            # Ensure to_dict returns ISO formatted strings
            d = f.to_dict()
            self.assertIsInstance(d["expiry"], str)
            self.assertIsInstance(d["timestamp"], str)


if __name__ == "__main__":
    unittest.main()