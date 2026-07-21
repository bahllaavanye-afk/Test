"""
Polymarket CLOB broker integration via py-clob-client.

Provides methods for market discovery, order book retrieval, order placement,
cancellation, and quoting for YES/NO binary markets on Polymarket.
"""

from typing import List, Dict, Any

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
    """Broker implementation for Polymarket's CLOB.

    Utilises ``py-clob-client`` to interact with Polymarket's order book,
    supporting binary YES/NO markets. All network calls are executed in a
    background thread to avoid blocking the event loop.
    """

    def __init__(self, private_key: str, chain_id: int = 137) -> None:
        """Create a new Polymarket broker instance.

        Args:
            private_key: Private key used for signing API requests.
            chain_id: Blockchain chain identifier (default is 137 for Polygon).
        """
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )

    async def get_markets(self, min_open_interest: float = 10000) -> List[Dict[str, Any]]:
        """Discover active markets with a minimum open interest.

        Args:
            min_open_interest: Minimum required open interest to include a market.

        Returns:
            A list of market dictionaries that meet the liquidity threshold.
            Returns an empty list on failure.
        """
        try:
            import asyncio
            markets = await asyncio.to_thread(self.client.get_markets)
            return [
                m for m in markets
                if float(m.get("openInterest", 0)) >= min_open_interest
            ]
        except Exception as e:
            logger.error("Polymarket market fetch failed", error=str(e))
            return []

    async def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """Retrieve the order book for a specific market token.

        Args:
            token_id: The market identifier used by Polymarket.

        Returns:
            A dictionary containing ``bids`` and ``asks`` lists.
        """
        import asyncio
        return await asyncio.to_thread(self.client.get_order_book, token_id)

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place a new order on Polymarket.

        Args:
            request: An :class:`OrderRequest` describing the order details.

        Returns:
            An :class:`OrderResult` with the broker's order identifier and status.

        Raises:
            BrokerError: If the underlying client raises any exception.
        """
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
            raise BrokerError(f"Polymarket: {e}")

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an existing order.

        Args:
            broker_order_id: The identifier of the order to cancel.

        Returns:
            ``True`` if cancellation succeeded, otherwise ``False``.
        """
        try:
            import asyncio
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            return True
        except Exception as e:
            logger.warning(
                "Polymarket cancel_order failed",
                order_id=broker_order_id,
                error=str(e)
            )
            return False

    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """Fetch details of a specific order.

        Args:
            broker_order_id: Identifier of the order to retrieve.

        Returns:
            A dictionary representing the order details.
        """
        import asyncio
        return await asyncio.to_thread(self.client.get_order, broker_order_id)

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Retrieve current positions.

        Polymarket does not expose position data through the CLOB client,
        so an empty list is returned.

        Returns:
            An empty list.
        """
        return []

    async def get_account(self) -> Dict[str, Any]:
        """Fetch account information.

        Returns:
            An empty dictionary as Polymarket does not provide account details
            via this client.
        """
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Generate a quote for a given market symbol.

        Args:
            symbol: Market token identifier.

        Returns:
            A :class:`QuoteResult` containing bid, ask, and last price.
        """
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

    async def get_historical(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical price data.

        Polymarket does not provide traditional OHLCV data, so an empty list
        is returned.

        Args:
            symbol: Market token identifier.
            interval: Desired time bucket (ignored).
            limit: Maximum number of records (ignored).

        Returns:
            An empty list.
        """
        return []