"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
"""
from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.exceptions import BrokerError
from app.utils.logging import logger
from app.config import settings

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    POLY_AVAILABLE = True
except ImportError:
    POLY_AVAILABLE = False


class PolymarketBroker(AbstractBroker):
    def __init__(self, private_key: str, chain_id: int = 137):
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )

    async def get_markets(self, min_open_interest: float = 10000) -> list[dict]:
        """Auto-discover active markets with sufficient liquidity."""
        try:
            import asyncio
            markets = await asyncio.to_thread(self.client.get_markets)
            return [
                m for m in markets
                if float(m.get("openInterest", 0)) >= min_open_interest
            ]
        except Exception as e:
            logger.exception(
                "Polymarket market fetch failed",
                error=str(e),
                min_open_interest=min_open_interest,
            )
            return []

    async def get_order_book(self, token_id: str) -> dict:
        """Retrieve the order book for a given market token."""
        try:
            import asyncio
            return await asyncio.to_thread(self.client.get_order_book, token_id)
        except Exception as e:
            logger.exception(
                "Polymarket get_order_book failed",
                token_id=token_id,
                error=str(e),
            )
            return {}

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place a new order on Polymarket."""
        try:
            import asyncio
            args = OrderArgs(
                token_id=request.symbol,
                price=request.limit_price or 0.5,
                size=request.quantity,
                side=request.side.upper(),
            )
            order = await asyncio.to_thread(self.client.create_and_post_order, args)
            return OrderResult(
                broker_order_id=str(order.get("orderID", "")),
                status=order.get("status", "pending"),
                raw_payload=order,
            )
        except Exception as e:
            logger.exception(
                "Polymarket place_order failed",
                symbol=request.symbol,
                quantity=request.quantity,
                side=request.side,
                error=str(e),
            )
            raise BrokerError(f"Polymarket: {e}") from e

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            import asyncio
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            return True
        except Exception as e:
            logger.exception(
                "Polymarket cancel_order failed",
                order_id=broker_order_id,
                error=str(e),
            )
            return False

    async def get_order(self, broker_order_id: str) -> dict:
        """Fetch details of a specific order."""
        try:
            import asyncio
            return await asyncio.to_thread(self.client.get_order, broker_order_id)
        except Exception as e:
            logger.exception(
                "Polymarket get_order failed",
                order_id=broker_order_id,
                error=str(e),
            )
            return {}

    async def get_positions(self) -> list[dict]:
        """Retrieve current positions; Polymarket does not expose positions via the client."""
        try:
            # Placeholder for future implementation; currently returns empty list.
            return []
        except Exception as e:
            logger.exception(
                "Polymarket get_positions failed",
                error=str(e),
            )
            return []

    async def get_account(self) -> dict:
        """Retrieve account information; Polymarket client does not expose account details."""
        try:
            # Placeholder for future implementation; currently returns empty dict.
            return {}
        except Exception as e:
            logger.exception(
                "Polymarket get_account failed",
                error=str(e),
            )
            return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Construct a quote from the best bid and ask."""
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        return QuoteResult(
            symbol=symbol,
            bid=best_bid,
            ask=best_ask,
            last=(best_bid + best_ask) / 2,
        )

    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[dict]:
        """Polymarket doesn't have traditional OHLCV; return empty list."""
        return []