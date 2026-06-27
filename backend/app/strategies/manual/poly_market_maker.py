"""Polymarket CLOB market making strategy."""
from __future__ import annotations

import pandas as pd

try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

CLOB_BASE = "https://clob.polymarket.com"


class PolymarketMarketMaker(AbstractStrategy):
    """
    Market make on liquid Polymarket CLOB markets.
    Posts bid/ask around mid with configurable spread.
    Zero fees for makers + PUSD rebates (as of 2025).

    Requires Polymarket API credentials to post orders (read-only analysis works without).
    """

    name = "poly_market_maker"
    display_name = "Polymarket CLOB Market Maker"
    market_type = "polymarket"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 10.0  # fast polling for CLOB

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = params or {}
        self.min_spread: float = float(p.get("min_spread", 0.02))
        self.max_position_pct: float = float(p.get("max_position_pct", 0.20))
        self.quote_size: float = float(p.get("quote_size", 10.0))
        self.min_volume: float = float(p.get("min_open_interest", 50_000.0))

    def description(self) -> str:
        return (
            f"Market make on liquid Polymarket CLOB markets (volume >${self.min_volume:,.0f}). "
            f"Min spread: {self.min_spread * 100:.1f}¢. Quote size: ${self.quote_size:.0f}. "
            "Zero maker fees + PUSD rebates. Source: Polymarket CLOB 2025."
        )

    async def _fetch_liquid_markets(self) -> list[dict]:
        if not _HTTPX:
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"{CLOB_BASE}/markets",
                    params={"limit": 50, "active": "true"},
                )
                r.raise_for_status()
                markets = r.json().get("data", [])
                return [
                    m for m in markets if float(m.get("volume", 0)) > self.min_volume
                ]
        except Exception:
            return []

    async def _fetch_order_book(self, token_id: str) -> dict:
        if not _HTTPX:
            return {}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"{CLOB_BASE}/book",
                    params={"token_id": token_id},
                )
                r.raise_for_status()
                return r.json()
        except Exception:
            return {}

    def _compute_spread(self, book: dict) -> tuple[float, float, float]:
        """Returns (best_bid, best_ask, mid)."""
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        if not bids or not asks:
            return 0.0, 1.0, 0.5
        best_bid = float(bids[0].get("price", 0))
        best_ask = float(asks[0].get("price", 1))
        mid = (best_bid + best_ask) / 2.0
        return best_bid, best_ask, mid

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Find liquid markets with wide enough spread to quote inside.
        data is not used directly — strategy fetches live CLOB order books.
        """
        markets = await self._fetch_liquid_markets()
        for market in markets:
            tokens = market.get("tokens", [])
            for token in tokens:
                if token.get("outcome", "").upper() != "YES":
                    continue
                token_id = token.get("token_id", "")
                book = await self._fetch_order_book(token_id)
                best_bid, best_ask, mid = self._compute_spread(book)
                current_spread = best_ask - best_bid

                if current_spread < self.min_spread:
                    continue

                # Ensure sufficient depth on both sides before posting
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
                ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])
                if min(bid_depth, ask_depth) < self.quote_size * 5:  # MUTATION: add depth filter to avoid thin books
                    continue

                # Post inside the spread to take priority
                our_bid = round(best_bid + self.min_spread / 2, 4)
                our_ask = round(best_ask - self.min_spread / 2, 4)

                # Create a signal to place the orders
                return Signal(
                    symbol=token_id,
                    side="both",
                    price_bid=our_bid,
                    price_ask=our_ask,
                    size=self.quote_size,
                    confidence=1.0,
                )
        return None
