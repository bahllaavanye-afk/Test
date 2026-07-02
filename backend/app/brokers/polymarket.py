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
        except (ValueError, TypeError) as e:
            logger.error(
                "Polymarket market fetch failed due to data conversion error",
                error=str(e),
                min_open_interest=min_open_interest,
            )
            return []
        except Exception as e:
            logger.exception(
                "Unexpected error while fetching Polymarket markets",
                error=str(e),
            )
            return []

    async def get_order_book(self, token_id: str) -> dict:
        """Retrieve the order book for a specific market token."""
        try:
            import asyncio
            return await asyncio.to_thread(self.client.get_order_book, token_id)
        except Exception as e:
            logger.exception(
                "Failed to get order book from Polymarket",
                token_id=token_id,
                error=str(e),
            )
            raise BrokerError(f"Polymarket order book error: {e}") from e

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order on Polymarket."""
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
        except (ValueError, TypeError) as e:
            logger.error(
                "Invalid order parameters for Polymarket",
                symbol=request.symbol,
                price=request.limit_price,
                size=request.quantity,
                side=request.side,
                error=str(e),
            )
            raise BrokerError(f"Polymarket: Invalid order parameters – {e}") from e
        except Exception as e:
            logger.exception(
                "Error placing order on Polymarket",
                symbol=request.symbol,
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
            logger.error(
                "Failed to cancel Polymarket order",
                broker_order_id=broker_order_id,
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
                "Error retrieving Polymarket order",
                broker_order_id=broker_order_id,
                error=str(e),
            )
            raise BrokerError(f"Polymarket get_order error: {e}") from e

    async def get_positions(self) -> list[dict]:
        """Polymarket does not expose positions via this client."""
        return []

    async def get_account(self) -> dict:
        """Polymarket does not expose account details via this client."""
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Generate a quote from the best bid/ask in the order book."""
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
        """Polymarket doesn't have traditional OHLCV data."""
        return []