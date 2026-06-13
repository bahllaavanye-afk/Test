"""Polymarket late-resolution arbitrage strategy."""
from __future__ import annotations

from datetime import UTC, datetime

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
    - Price trending up over last 6h
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

    def description(self) -> str:
        return (
            f"Buy near-certain (>{self.min_price * 100:.0f}%) YES contracts within "
            f"{self.max_hours}h of resolution. Expected return: (1 - price) / price. "
            "Source: Polymarket CLOB late-resolution arbitrage."
        )

    def _hours_to_resolution(self, end_date_str: str) -> float:
        try:
            end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            now = datetime.now(UTC)
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
        data is not used directly — strategy fetches live CLOB data.
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

                token_id = token.get("token_id", "")
                history = await self._fetch_price_history(token_id)
                if len(history) >= 2:
                    old_price = float(history[0].get("p", price))
                    trend = price - old_price
                    if trend < self.min_trend:
                        continue

                expected_return = (1.0 - price) / price
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
                    },
                )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Proxy backtest: buy when close > min_price (near-certain YES contract proxy).
        In real use, this strategy is live-only (resolution date is the key input).
        """
        false_series = pd.Series(False, index=df.index)
        default = BacktestSignals(
            entries=false_series,
            exits=false_series,
        )

        if "close" not in df.columns or len(df) < 2:
            return default

        close = df["close"].astype(float)
        entries = (close.shift(1) > self.min_price).fillna(False).astype(bool)
        exits = (close.shift(1) >= 0.99).fillna(False).astype(bool)

        return BacktestSignals(
            entries=entries,
            exits=exits,
        )
