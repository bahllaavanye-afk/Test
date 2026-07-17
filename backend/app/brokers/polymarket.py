"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
"""
from numbers import Real
from typing import Any

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
        """
        Initialize the Polymarket broker client.

        Args:
            private_key: Private key string for authentication; must be non‑empty.
            chain_id: Blockchain chain identifier; must be a positive integer.

        Raises:
            ValueError: If inputs are invalid.
            ImportError: If the required py-clob-client library is missing.
        """
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        if not isinstance(private_key, str) or not private_key.strip():
            raise ValueError("private_key must be a non-empty string")
        if not isinstance(chain_id, int) or chain_id <= 0:
            raise ValueError("chain_id must be a positive integer")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )

    async def get_markets(self, min_open_interest: float = 10000) -> list[dict]:
        """
        Retrieve active markets with a minimum open interest.

        Args:
            min_open_interest: Minimum open interest threshold; must be non‑negative.

        Returns:
            List of market dictionaries meeting the liquidity criteria.

        Raises:
            ValueError: If min_open_interest is negative or not a number.
        """
        if not isinstance(min_open_interest, Real):
            raise ValueError("min_open_interest must be a numeric value")
        if min_open_interest < 0:
            raise ValueError("min_open_interest cannot be negative")
        try:
            import asyncio
            markets = await asyncio.to_thread(self.client.get_markets)
            return [
                m for m in markets
                if float(m.get("openInterest", 0)) >= float(min_open_interest)
            ]
        except Exception as e:
            logger.error("Polymarket market fetch failed", error=str(e))
            return []

    async def get_order_book(self, token_id: str) -> dict:
        """
        Fetch the order book for a given market token.

        Args:
            token_id: Market token identifier; must be a non‑empty string.

        Returns:
            Order book dictionary.

        Raises:
            ValueError: If token_id is invalid.
        """
        if not isinstance(token_id, str) or not token_id.strip():
            raise ValueError("token_id must be a non-empty string")
        import asyncio
        return await asyncio.to_thread(self.client.get_order_book, token_id)

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Submit a new order to Polymarket.

        Args:
            request: OrderRequest instance containing order details.

        Returns:
            OrderResult encapsulating the broker response.

        Raises:
            ValueError: If request or its fields are invalid.
            BrokerError: If the underlying client raises an exception.
        """
        if not isinstance(request, OrderRequest):
            raise ValueError("request must be an instance of OrderRequest")
        if not isinstance(request.symbol, str) or not request.symbol.strip():
            raise ValueError("request.symbol must be a non-empty string")
        if request.limit_price is not None:
            if not isinstance(request.limit_price, Real):
                raise ValueError("request.limit_price must be a numeric value")
            if request.limit_price <= 0:
                raise ValueError("request.limit_price must be positive")
        if not isinstance(request.quantity, Real) or request.quantity <= 0:
            raise ValueError("request.quantity must be a positive number")
        if not isinstance(request.side, str) or request.side.upper() not in {"BUY", "SELL"}:
            raise ValueError("request.side must be either 'BUY' or 'SELL' (case‑insensitive)")

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
        """
        Cancel an existing order.

        Args:
            broker_order_id: Identifier of the order to cancel; must be a non‑empty string.

        Returns:
            True if cancellation succeeded, False otherwise.

        Raises:
            ValueError: If broker_order_id is invalid.
        """
        if not isinstance(broker_order_id, str) or not broker_order_id.strip():
            raise ValueError("broker_order_id must be a non-empty string")
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

    async def get_order(self, broker_order_id: str) -> dict:
        """
        Retrieve details of a specific order.

        Args:
            broker_order_id: Order identifier; must be a non‑empty string.

        Returns:
            Order detail dictionary.

        Raises:
            ValueError: If broker_order_id is invalid.
        """
        if not isinstance(broker_order_id, str) or not broker_order_id.strip():
            raise ValueError("broker_order_id must be a non-empty string")
        import asyncio
        return await asyncio.to_thread(self.client.get_order, broker_order_id)

    async def get_positions(self) -> list[dict]:
        """
        Retrieve current positions.

        Returns:
            Empty list as Polymarket does not expose positions via this API.
        """
        return []

    async def get_account(self) -> dict:
        """
        Retrieve account information.

        Returns:
            Empty dict as Polymarket does not expose account details via this API.
        """
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        """
        Obtain the best bid/ask quote for a market.

        Args:
            symbol: Market symbol/token; must be a non‑empty string.

        Returns:
            QuoteResult containing bid, ask, and last price.

        Raises:
            ValueError: If symbol is invalid.
        """
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
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
        """
        Retrieve historical data for a market.

        Args:
            symbol: Market symbol/token; must be a non‑empty string.
            interval: Time interval string (e.g., "1d", "1h"); must be non‑empty.
            limit: Maximum number of data points; must be a positive integer.

        Returns:
            Empty list as Polymarket does not provide traditional OHLCV data.

        Raises:
            ValueError: If any argument is invalid.
        """
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(interval, str) or not interval.strip():
            raise ValueError("interval must be a non-empty string")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        return []  # Polymarket doesn't have traditional OHLCV