"""
Crypto Perpetual Basis / Funding Rate Carry Strategy.

When BTC perpetual futures funding rate > 0.01% per 8h (annualised ~10.95%),
longs are paying shorts. Capture this by:
  - Short BTC perpetual (receive funding)
  - Long BTC spot (hedge delta)

Net P&L ≈ funding rate × notional, delta-neutral.

Data source: Binance public REST API (free, unauthenticated).
  GET https://fapi.binance.com/fapi/v1/fundingRate
  GET https://fapi.binance.com/fapi/v1/premiumIndex
"""
import json
import urllib.request

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

BINANCE_FUNDING_URL = (
    "https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=10"
)
BINANCE_PREMIUM_URL = (
    "https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
)


def _fetch_funding_rate(symbol: str = "BTCUSDT") -> float | None:
    """Fetch latest 8-hour funding rate from Binance public REST."""
    try:
        url = BINANCE_PREMIUM_URL.format(symbol=symbol)
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        return float(data.get("lastFundingRate", 0))
    except Exception:
        return None


class CryptoBasisRollStrategy(AbstractStrategy):
    name = "crypto_basis_roll"
    display_name = "Crypto Perpetual Basis Roll (BTC Funding)"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 3600.0  # check hourly (funding settles every 8h)

    FUNDING_THRESHOLD = 0.0001  # 0.01% per 8h ≈ ~11% annualized
    FUNDING_UNWIND = 0.00005   # unwind when funding falls below 0.005%

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.funding_threshold = p.get("funding_threshold", self.FUNDING_THRESHOLD)
        self.funding_unwind = p.get("funding_unwind", self.FUNDING_UNWIND)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in data.columns:
            return None

        # Try live funding rate from Binance
        perp_symbol = symbol.replace("-", "").replace("/", "") + "T"
        if not perp_symbol.endswith("USDT"):
            perp_symbol = "BTCUSDT"

        live_rate = _fetch_funding_rate(perp_symbol)

        if live_rate is not None:
            funding_rate = live_rate
        elif "funding_rate" in data.columns:
            funding_rate = float(data["funding_rate"].iloc[-1])
        else:
            return None

        if funding_rate > self.funding_threshold:
            # Positive funding: longs pay shorts → short perp + long spot
            carry_ann = funding_rate * 3 * 365  # 3 settlements/day × 365
            confidence = min(0.85, 0.60 + funding_rate * 1000)
            return Signal(
                symbol=symbol,
                side="sell",   # sell = short perpetual (sell dearer)
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "funding_rate_8h": round(funding_rate, 6),
                    "carry_ann_pct": round(carry_ann * 100, 2),
                    "trade_type": "short_perp_long_spot",
                },
            )
        elif funding_rate < -self.funding_threshold:
            # Negative funding: shorts pay longs → long perp + short spot
            confidence = min(0.85, 0.60 + abs(funding_rate) * 1000)
            return Signal(
                symbol=symbol,
                side="buy",    # buy = long perpetual
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "funding_rate_8h": round(funding_rate, 6),
                    "trade_type": "long_perp_short_spot",
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "funding_rate" in df.columns:
            fr = df["funding_rate"].shift(1)
        else:
            # Proxy: when price trend is strong, funding is typically positive
            ret = df["close"].pct_change()
            fr = ret.rolling(8).mean() * 0.01  # crude proxy

        entries = fr < -self.funding_threshold       # negative funding → long perp
        exits = fr >= -self.funding_unwind
        short_entries = fr > self.funding_threshold  # positive funding → short perp
        short_exits = fr <= self.funding_unwind

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )
