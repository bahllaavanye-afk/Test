"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
Provides enhanced entry/exit validation to improve signal quality.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.exceptions import BrokerError
from app.utils.logging import logger
from app.config import settings

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    POLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    POLY_AVAILABLE = False


class PolymarketBroker(AbstractBroker):
    """Concrete broker for Polymarket CLOB."""

    def __init__(self, private_key: str, chain_id: int = 137) -> None:
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )

    # --------------------------------------------------------------------- #
    # Market discovery & order‑book utilities
    # --------------------------------------------------------------------- #
    async def get_markets(self, min_open_interest: float = 10000) -> List[Dict[str, Any]]:
        """Return active markets filtered by a minimum open interest."""
        try:
            import asyncio
            markets = await asyncio.to_thread(self.client.get_markets)
            filtered = [
                m for m in markets
                if float(m.get("openInterest", 0)) >= min_open_interest
            ]
            logger.debug(
                "Polymarket markets fetched",
                total=len(markets),
                filtered=len(filtered),
                min_oi=min_open_interest,
            )
            return filtered
        except Exception as e:  # pragma: no cover
            logger.error("Polymarket market fetch failed", error=str(e))
            return []

    async def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """Retrieve the raw order book for a given token."""
        import asyncio
        return await asyncio.to_thread(self.client.get_order_book, token_id)

    # --------------------------------------------------------------------- #
    # Quote handling with enhanced validation
    # --------------------------------------------------------------------- #
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return best bid/ask and a mid price with basic liquidity checks."""
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])

        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0

        # Compute weighted mid price using top 5 levels if available
        mid_price = self._weighted_mid_price(bids, asks)

        # Simple volume filter – log if depth is thin
        bid_volume = sum(float(b["size"]) for b in bids[:5])
        ask_volume = sum(float(a["size"]) for a in asks[:5])
        if bid_volume < settings.min_liquidity or ask_volume < settings.min_liquidity:
            logger.warning(
                "Low liquidity detected",
                symbol=symbol,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
            )

        return QuoteResult(
            symbol=symbol,
            bid=best_bid,
            ask=best_ask,
            last=mid_price,
        )

    # --------------------------------------------------------------------- #
    # Order lifecycle
    # --------------------------------------------------------------------- #
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order after confirming spread and depth constraints."""
        try:
            import asyncio

            # Validate market conditions before sending the order
            quote = await self.get_quote(request.symbol)
            if not self._is_spread_acceptable(quote):
                raise BrokerError(
                    f"Spread too wide for {request.symbol}: "
                    f"bid={quote.bid:.4f} ask={quote.ask:.4f}"
                )
            if not self._has_sufficient_depth(request.symbol):
                raise BrokerError(f"Insufficient depth for {request.symbol}")

            args = OrderArgs(
                token_id=request.symbol,
                price=request.limit_price or (quote.bid + quote.ask) / 2,
                size=request.quantity,
                side=request.side.upper(),
            )
            order = await asyncio.to_thread(self.client.create_and_post_order, args)
            logger.info(
                "Polymarket order placed",
                symbol=request.symbol,
                side=request.side,
                price=args.price,
                size=args.size,
                broker_order_id=order.get("orderID"),
            )
            return OrderResult(
                broker_order_id=str(order.get("orderID", "")),
                status=order.get("status", "pending"),
                raw_payload=order,
            )
        except BrokerError:
            raise
        except Exception as e:  # pragma: no cover
            raise BrokerError(f"Polymarket: {e}") from e

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            import asyncio
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            logger.info("Polymarket order cancelled", order_id=broker_order_id)
            return True
        except Exception as e:  # pragma: no cover
            logger.warning(
                "Polymarket cancel_order failed",
                order_id=broker_order_id,
                error=str(e),
            )
            return False

    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """Fetch order details."""
        import asyncio
        return await asyncio.to_thread(self.client.get_order, broker_order_id)

    # --------------------------------------------------------------------- #
    # Portfolio helpers (placeholders – Polymarket does not expose them)
    # --------------------------------------------------------------------- #
    async def get_positions(self) -> List[Dict[str, Any]]:
        return []

    async def get_account(self) -> Dict[str, Any]:
        return {}

    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> List[Dict[str, Any]]:
        """Polymarket does not provide traditional OHLCV data."""
        return []

    # --------------------------------------------------------------------- #
    # Internal validation utilities
    # --------------------------------------------------------------------- #
    def _weighted_mid_price(
        self,
        bids: List[Dict[str, Any]],
        asks: List[Dict[str, Any]],
        depth: int = 5,
    ) -> float:
        """Calculate a volume‑weighted mid price from top order‑book levels."""
        if not bids or not asks:
            return 0.5  # fallback for empty book

        top_bids = bids[:depth]
        top_asks = asks[:depth]

        bid_weight = sum(float(b["size"]) for b in top_bids)
        ask_weight = sum(float(a["size"]) for a in top_asks)

        weighted_bid = sum(float(b["price"]) * float(b["size"]) for b in top_bids) / max(bid_weight, 1e-9)
        weighted_ask = sum(float(a["price"]) * float(a["size"]) for a in top_asks) / max(ask_weight, 1e-9)

        return (weighted_bid + weighted_ask) / 2.0

    def _is_spread_acceptable(self, quote: QuoteResult, max_spread: Optional[float] = None) -> bool:
        """Determine if the current spread meets strategy thresholds."""
        max_spread = max_spread if max_spread is not None else getattr(settings, "max_spread", 0.05)
        spread = quote.ask - quote.bid
        acceptable = spread <= max_spread
        if not acceptable:
            logger.debug(
                "Spread check failed",
                symbol=quote.symbol,
                spread=spread,
                max_allowed=max_spread,
            )
        return acceptable

    def _has_sufficient_depth(self, symbol: str, min_volume: Optional[float] = None) -> bool:
        """Verify that both sides of the book have at least `min_volume` liquidity."""
        min_volume = min_volume if min_volume is not None else getattr(settings, "min_liquidity", 10.0)
        ob = asyncio.run(self.get_order_book(symbol))  # Synchronous call for validation
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        bid_vol = sum(float(b["size"]) for b in bids[:5])
        ask_vol = sum(float(a["size"]) for a in asks[:5])
        sufficient = bid_vol >= min_volume and ask_vol >= min_volume
        if not sufficient:
            logger.debug(
                "Depth check failed",
                symbol=symbol,
                bid_vol=bid_vol,
                ask_vol=ask_vol,
                required=min_volume,
            )
        return sufficient

    # --------------------------------------------------------------------- #
    # Exit logic helper
    # --------------------------------------------------------------------- #
    async def calculate_optimal_exit(self, symbol: str, target_profit: float) -> Optional[float]:
        """
        Compute a price level for exiting a position that satisfies a target profit
        while respecting current market depth. Returns ``None`` if an exit price
        cannot be determined under the current order‑book conditions.
        """
        quote = await self.get_quote(symbol)
        if quote.bid == 0 or quote.ask == 0:
            return None

        # Simple profit target based on mid price
        desired_price = quote.last * (1 + target_profit)
        # Ensure the desired price is reachable within the ask side depth
        ob = await self.get_order_book(symbol)
        asks = ob.get("asks", [])
        cumulative = 0.0
        for level in asks:
            level_price = float(level["price"])
            level_size = float(level["size"])
            cumulative += level_size
            if level_price <= desired_price:
                return level_price
        # If not reachable, fallback to best ask
        return quote.ask if quote.ask > desired_price else None