"""Polymarket late-resolution arbitrage strategy."""
from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

try:
    import httpx

    _HTTPX = True
except ImportError:
    _HTTPX = False

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

CLOB_BASE = "https://clob.polymarket.com"


class PolymarketLateResolution(AbstractStrategy):
    """
    Buy Polymarket YES contracts that are nearly certain (>80%) with
    <48h to resolution. Expected return: (1.0 - price) / price in <48h.

    Heuristic for 'nearly certain':
    - YES price > min_price (default 0.80)
    - Price trending up over last 6h with at least min_trend
    - Expected return above min_expected_return
    - Sufficient market volume (min_volume)
    - Time to resolution < max_hours_to_resolution (default 48h)
    """

    name = "poly_late_resolution"
    display_name = "Polymarket Late-Resolution Arbitrage"
    market_type = "polymarket"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 300.0  # poll every 5 minutes

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = params or {}
        self.min_price: float = float(p.get("min_price", 0.80))
        self.max_hours: int = int(p.get("max_hours_to_resolution", 48))
        self.min_trend: float = float(p.get("min_price_trend", 0.02))
        self.min_expected_return: float = float(p.get("min_expected_return", 0.05))  # 5%
        self.min_volume: float = float(p.get("min_volume", 0.0))

    def description(self) -> str:
        return (
            f"Buy near-certain (>{self.min_price * 100:.0f}%) YES contracts within "
            f"{self.max_hours}h of resolution. Expected return: (1 - price) / price. "
            "Source: Polymarket CLOB late-resolution arbitrage."
        )

    def _hours_to_resolution(self, end_date_str: str) -> float:
        try:
            end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (end - now).total_seconds() / 3600.0
        except Exception:
            return 9999.0

    async def _fetch_markets(self) -> list[dict]:
        if not _HTTPX:
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"{CLOB_BASE}/markets",
                    params={"limit": 100, "active": "true"},
                )
                r.raise_for_status()
                return r.json().get("data", [])
        except Exception:
            return []

    async def _fetch_price_history(self, token_id: str) -> list[dict]:
        if not _HTTPX:
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"{CLOB_BASE}/prices-history",
                    params={"token_id": token_id, "interval": "6h", "fidelity": 60},
                )
                r.raise_for_status()
                return r.json().get("history", [])
        except Exception:
            return []

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Scan Polymarket markets for near-certain contracts close to resolution.
        `data` is not used directly — strategy fetches live CLOB data.
        """
        markets = await self._fetch_markets()
        for market in markets:
            end_date = market.get("end_date_iso") or market.get("end_date", "")
            if not end_date:
                continue
            hours_left = self._hours_to_resolution(end_date)
            if hours_left <= 0 or hours_left > self.max_hours:
                continue

            tokens = market.get("tokens", [])
            for token in tokens:
                if token.get("outcome", "").upper() != "YES":
                    continue

                price = float(token.get("price", 0))
                if price < self.min_price or price >= 0.99:
                    continue

                # Volume filter
                volume = float(token.get("volume", 0))
                if volume < self.min_volume:
                    continue

                token_id = token.get("token_id", "")
                history = await self._fetch_price_history(token_id)
                if len(history) >= 2:
                    # Extract price series
                    prices = [float(entry.get("p", price)) for entry in history]
                    price_start = prices[0]
                    price_end = prices[-1]
                    trend = price_end - price_start
                    avg_price = sum(prices) / len(prices)
                    if trend < self.min_trend or price <= avg_price:
                        continue

                expected_return = (1.0 - price) / price
                if expected_return < self.min_expected_return:
                    continue

                return Signal(
                    strategy_name=self.name,
                    strategy_type=self.strategy_type,
                    risk_bucket=self.risk_bucket,
                    symbol=market.get("question", "POLY_MARKET"),
                    side="buy",
                    confidence=price,
                    metadata={
                        "market_id": market.get("condition_id"),
                        "token_id": token_id,
                        "price": price,
                        "hours_to_resolution": round(hours_left, 2),
                        "expected_return_pct": round(expected_return * 100, 2),
                        "order_type": "limit",
                        "volume": volume,
                    },
                )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Proxy backtest: buy when close > min_price and expected return > min_expected_return.
        Exit when close >= 0.99 or close falls below min_price.
        """
        false_series = pd.Series(False, index=df.index)
        default = BacktestSignals(
            entries=false_series,
            exits=false_series,
        )

        required_cols = {"close"}
        if not required_cols.issubset(df.columns) or len(df) < 2:
            return default

        close = df["close"].astype(float)

        # Expected return based on price
        expected_return = (1.0 - close) / close

        entries = (
            (close.shift(1) > self.min_price)
            & (expected_return.shift(1) > self.min_expected_return)
        ).fillna(False).astype(bool)

        exits = (
            (close.shift(1) >= 0.99) | (close.shift(1) < self.min_price)
        ).fillna(False).astype(bool)

        return BacktestSignals(
            entries=entries,
            exits=exits,
        )