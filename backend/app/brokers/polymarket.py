"""
Polymarket CLOB broker integration via py-clob-client.

Provides methods to discover markets, retrieve order books, place and cancel orders,
and obtain quotes for binary YES/NO markets on Polymarket. The implementation
relies on asynchronous wrappers around the synchronous py-clob-client library.
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
    """Broker implementation for Polymarket's CLOB using `py-clob-client`.

    The broker supports binary YES/NO markets and provides a thin async wrapper
    around the underlying client. All network‑bound calls are executed in a
    thread pool via ``asyncio.to_thread`` to avoid blocking the event loop.
    """

    def __init__(self, private_key: str, chain_id: int = 137) -> None:
        """Create a new Polymarket broker instance.

        Args:
            private_key: Private key used for signing requests to Polymarket.
            chain_id: Ethereum chain identifier; defaults to 137 (Polygon).

        Raises:
            ImportError: If the required ``py-clob-client`` package is unavailable.
        """
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )

    async def get_markets(self, min_open_interest: float = 10000) -> list[dict]:
        """Retrieve active markets with a minimum open interest.

        Args:
            min_open_interest: Minimum open interest threshold to filter markets.

        Returns:
            A list of market dictionaries that meet the liquidity requirement.
            Returns an empty list if the fetch fails.
        """
        try:
            import asyncio
            markets = await asyncio.to_thread(self.client.get_markets)
            return [m for m in markets if float(m.get("openInterest", 0)) >= min_open_interest]
        except Exception as e:
            logger.error("Polymarket market fetch failed", error=str(e))
            return []

    async def get_order_book(self, token_id: str) -> dict:
        """Fetch the order book for a given market token.

        Args:
            token_id: Identifier of the market token.

        Returns:
            The raw order book dictionary as returned by the Polymarket client.
        """
        import asyncio
        return await asyncio.to_thread(self.client.get_order_book, token_id)

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place a new order on Polymarket.

        Args:
            request: An ``OrderRequest`` containing symbol, quantity, side, and
                optional limit price.

        Returns:
            An ``OrderResult`` encapsulating the broker order ID, status, and
            raw payload from the exchange.

        Raises:
            BrokerError: If the underlying client raises an exception.
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
            ``True`` if the cancellation succeeded, ``False`` otherwise.
        """
        try:
            import asyncio
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            return True
        except Exception as e:
            logger.warning(
                "Polymarket cancel_order failed",
                order_id=broker_order_id,
                error=str(e),
            )
            return False

    async def get_order(self, broker_order_id: str) -> dict:
        """Retrieve details of a specific order.

        Args:
            broker_order_id: The broker‑assigned order identifier.

        Returns:
            The raw order dictionary from Polymarket.
        """
        import asyncio
        return await asyncio.to_thread(self.client.get_order, broker_order_id)

    async def get_positions(self) -> list[dict]:
        """Return current positions.

        Polymarket does not expose traditional position data, so an empty list is
        returned.

        Returns:
            An empty list.
        """
        return []

    async def get_account(self) -> dict:
        """Fetch account information.

        Polymarket does not provide a detailed account endpoint in this client,
        so an empty dictionary is returned.

        Returns:
            An empty dictionary.
        """
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Obtain the best bid/ask quote for a market.

        Args:
            symbol: Market token identifier.

        Returns:
            A ``QuoteResult`` containing bid, ask, and last price derived from the
            order book.
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
    ) -> list[dict]:
        """Retrieve historical price data.

        Polymarket does not provide traditional OHLCV data, so an empty list is
        returned.

        Args:
            symbol: Market token identifier.
            interval: Desired time bucket (ignored).
            limit: Maximum number of data points (ignored).

        Returns:
            An empty list.
        """
        return []