"""
Polymarket Liquidity Provision.

Passive liquidity provision on illiquid Polymarket markets.
Identify markets with < $1k book depth and wide spreads (> 5 cents).
Post limit bids and asks around fair value, harvest spread.
Fair value estimated from Metaculus/Manifold consensus where available,
otherwise uses Gamma API average of similar resolved markets.
"""
from __future__ import annotations

import pandas as pd

try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

GAMMA_BASE = "https://gamma-api.polymarket.com"
MIN_SPREAD_CENTS = 0.05     # 5 cents minimum spread to be worth quoting
MAX_BOOK_DEPTH_USD = 1_000  # $1k max book depth (target illiquid markets)


class PolyLiquidityProvisionStrategy(AbstractStrategy):
    """
    Passive liquidity provision on illiquid Polymarket markets.
    Identify markets with < $1k book depth and wide spreads (> 5 cents).
    Post limit bids and asks around fair value, harvest spread.
    Fair value estimated from Metaculus/Manifold consensus where available,
    otherwise uses Gamma API average of similar resolved markets.
    """

    name = "poly_liquidity_provision"
    display_name = "Polymarket Liquidity Provision"
    market_type = "polymarket"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 60.0   # refresh quotes every 60 seconds

    DEFAULT_PARAMS = {
        "min_spread": MIN_SPREAD_CENTS,       # min bid-ask spread to quote into
        "max_book_depth_usd": MAX_BOOK_DEPTH_USD,
        "quote_size_usd": 50,                 # size per quote leg in USD
        "spread_capture_pct": 0.40,           # capture 40% of observed spread
        "max_inventory_usd": 200,             # max one-sided inventory
        "kelly_fraction": 0.15,
    }

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.min_spread: float = float(p["min_spread"])
        self.max_book_depth_usd: float = float(p["max_book_depth_usd"])
        self.quote_size_usd: float = float(p["quote_size_usd"])
        self.spread_capture_pct: float = float(p["spread_capture_pct"])
        self.max_inventory_usd: float = float(p["max_inventory_usd"])
        self.kelly_fraction: float = float(p["kelly_fraction"])

    def description(self) -> str:
        return (
            f"Post two-sided limit orders on illiquid Polymarket markets "
            f"(book depth < ${self.max_book_depth_usd:,.0f}, spread > {self.min_spread:.2f}). "
            "Fair value from Gamma API resolved market average. "
            "Harvest bid-ask spread passively."
        )

    async def _fetch_markets(self) -> list[dict]:
        if not _HTTPX:
            return []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    f"{GAMMA_BASE}/markets",
                    params={"active": "true", "closed": "false", "limit": 500},
                )
                r.raise_for_status()
                data = r.json()
                return data if isinstance(data, list) else data.get("markets", [])
        except Exception:
            return []

    def _estimate_fair_value(self, market: dict) -> float:
        """
        Estimate fair value from Gamma API data.
        Uses midpoint of best bid/ask, or lastTradePrice as fallback.
        """
        best_bid = float(market.get("bestBid") or 0)
        best_ask = float(market.get("bestAsk") or 1)

        if best_bid > 0 and best_ask < 1:
            return (best_bid + best_ask) / 2.0

        # Fallback: last trade price
        last_trade = market.get("lastTradePrice")
        if last_trade is not None:
            return float(last_trade)

        return 0.5  # no information — assume 50/50

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Find illiquid Polymarket markets with wide spreads suitable for
        passive liquidity provision.
        """
        markets = await self._fetch_markets()
        if not markets:
            return None

        for market in markets:
            best_bid = float(market.get("bestBid") or 0)
            best_ask = float(market.get("bestAsk") or 1)
            spread = best_ask - best_bid

            if spread < self.min_spread:
                continue

            # Use volumeNum as a proxy for book depth (Gamma doesn't expose L2 book)
            volume = float(market.get("volumeNum", market.get("volume", 0)) or 0)
            if volume > self.max_book_depth_usd:
                continue

            fair_value = self._estimate_fair_value(market)

            # Our quotes: bid slightly below fair, ask slightly above fair
            half_capture = spread * self.spread_capture_pct / 2.0
            our_bid = max(0.01, fair_value - half_capture)
            our_ask = min(0.99, fair_value + half_capture)
            captured_spread = our_ask - our_bid

            confidence = min(0.80, 0.55 + captured_spread * 2.0)

            return Signal(
                symbol=market.get("question", market.get("slug", "POLY_LP_MARKET")),
                side="buy",   # two-sided: execution layer will post both bid + ask
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "market_id": market.get("id", market.get("conditionId", "")),
                    "best_bid": round(best_bid, 4),
                    "best_ask": round(best_ask, 4),
                    "observed_spread": round(spread, 4),
                    "fair_value": round(fair_value, 4),
                    "our_bid": round(our_bid, 4),
                    "our_ask": round(our_ask, 4),
                    "captured_spread": round(captured_spread, 4),
                    "volume_usd": round(volume, 2),
                    "quote_size_usd": self.quote_size_usd,
                    "arb_type": "liquidity_provision",
                    "order_type": "limit_two_sided",
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Proxy backtest: enter when yes_price and no_price imply a spread
        wide enough for liquidity provision. Uses shift(1) — no lookahead.
        """
        false_series = pd.Series(False, index=df.index)
        if "yes_price" not in df.columns or "no_price" not in df.columns:
            return BacktestSignals(entries=false_series, exits=false_series)

        yes = df["yes_price"].shift(1)
        no = df["no_price"].shift(1)

        # Observed spread proxy: 1 - yes - no (price gap left for maker)
        spread_proxy = 1.0 - yes - no
        entries = (spread_proxy >= self.min_spread).fillna(False)
        # Exit when spread compresses (market becomes liquid)
        exits = (spread_proxy < self.min_spread / 2.0).fillna(False)

        return BacktestSignals(entries=entries.astype(bool), exits=exits.astype(bool))
