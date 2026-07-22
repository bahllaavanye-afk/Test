"""
Polymarket CLOB broker integration via py-clob-client.

This module provides a concrete implementation of the ``AbstractBroker`` interface
for Polymarket's Central Limit Order Book (CLOB). It supports YES/NO binary market
trading, basic entry validation, and simple exit confirmation. The broker does not
expose positions or account balances through the client API, and those methods
return empty structures accordingly.
"""

from typing import List, Dict, Optional

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
    """Broker implementation for Polymarket CLOB.

    The broker validates entry conditions based on spread and slippage tolerances
    before submitting orders. It also performs a lightweight poll after order
    creation to capture rapid status changes.
    """

    def __init__(self, private_key: str, chain_id: int = 137) -> None:
        """Create a new ``PolymarketBroker`` instance.

        Args:
            private_key: Private key used to sign API requests.
            chain_id: Blockchain chain identifier (default is 137 for Polygon).
        """
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client: ClobClient = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )
        # Configurable thresholds for entry validation
        self.max_spread: float = getattr(settings, "POLY_MAX_SPREAD", 0.02)  # 2% default
        self.price_slippage_tolerance: float = getattr(
            settings, "POLY_SLIPPAGE_TOLERANCE", 0.01
        )  # 1% default

    async def get_markets(self, min_open_interest: float = 10000) -> List[Dict]:
        """Discover active markets that meet a minimum open‑interest threshold.

        Args:
            min_open_interest: Minimum open interest required for a market to be returned.

        Returns:
            A list of market dictionaries filtered by ``min_open_interest``.
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

    async def get_order_book(self, token_id: str) -> Dict:
        """Retrieve the current order book for a given market token.

        Args:
            token_id: The market token identifier.

        Returns:
            The raw order‑book dictionary from the Polymarket client.
        """
        import asyncio
        return await asyncio.to_thread(self.client.get_order_book, token_id)

    async def _validate_entry(self, symbol: str, limit_price: Optional[float]) -> bool:
        """Validate tighter entry criteria for a market.

        The validation checks that:
        * The spread between the best bid and ask is below ``self.max_spread``.
        * If a ``limit_price`` is supplied, it lies within
          ``self.price_slippage_tolerance`` of the mid price.

        Args:
            symbol: Market token identifier.
            limit_price: Optional price limit supplied by the caller.

        Returns:
            ``True`` if the market satisfies the entry criteria; otherwise ``False``.
        """
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            logger.debug("Order book empty for symbol", symbol=symbol)
            return False

        best_bid = float(bids[0].get("price", 0))
        best_ask = float(asks[0].get("price", 1))
        spread = best_ask - best_bid
        if spread < 0:
            logger.warning("Negative spread detected", symbol=symbol, spread=spread)
            return False
        if spread > self.max_spread:
            logger.debug(
                "Spread exceeds threshold",
                symbol=symbol,
                spread=spread,
                max_spread=self.max_spread,
            )
            return False

        mid_price = (best_bid + best_ask) / 2
        if limit_price is not None:
            deviation = abs(limit_price - mid_price) / mid_price
            if deviation > self.price_slippage_tolerance:
                logger.debug(
                    "Limit price deviates too much from mid price",
                    symbol=symbol,
                    limit_price=limit_price,
                    mid_price=mid_price,
                    deviation=deviation,
                    tolerance=self.price_slippage_tolerance,
                )
                return False
        return True

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order after confirming entry conditions.

        Args:
            request: An ``OrderRequest`` containing symbol, quantity, side, and optional limit price.

        Returns:
            An ``OrderResult`` encapsulating the broker order identifier, final status, and raw payload.

        Raises:
            BrokerError: If validation fails or the order cannot be placed.
        """
        try:
            import asyncio

            # Validate entry conditions before sending the order
            is_valid = await self._validate_entry(
                symbol=request.symbol,
                limit_price=request.limit_price,
            )
            if not is_valid:
                raise BrokerError(
                    f"Polymarket: Entry validation failed for {request.symbol}"
                )

            args = OrderArgs(
                token_id=request.symbol,
                price=request.limit_price or 0.5,
                size=request.quantity,
                side=request.side.upper(),
            )
            order = await asyncio.to_thread(self.client.create_and_post_order, args)

            # Basic exit confirmation: ensure order status is final before returning
            status = order.get("status", "pending")
            if status not in {"filled", "canceled", "rejected"}:
                # Poll once more to capture any rapid status change
                import time

                time.sleep(0.2)
                refreshed = await self.get_order(order.get("orderID", ""))
                status = refreshed.get("status", status)

            return OrderResult(
                broker_order_id=str(order.get("orderID", "")),
                status=status,
                raw_payload=order,
            )
        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Polymarket: {e}")

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an existing order.

        Args:
            broker_order_id: The identifier of the order to cancel.

        Returns:
            ``True`` if the cancellation request succeeded; otherwise ``False``.
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

    async def get_order(self, broker_order_id: str) -> Dict:
        """Retrieve a specific order's details.

        Args:
            broker_order_id: The broker‑specific order identifier.

        Returns:
            A dictionary with the order's details as returned by the client.
        """
        import asyncio
        return await asyncio.to_thread(self.client.get_order, broker_order_id)

    async def get_positions(self) -> List[Dict]:
        """Polymarket does not expose positions via the CLOB client.

        Returns:
            An empty list, indicating that position data is unavailable.
        """
        return []

    async def get_account(self) -> Dict:
        """Polymarket does not expose account balances via the CLOB client.

        Returns:
            An empty dictionary, indicating that account information is unavailable.
        """
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return the best bid, ask, and mid price for a market.

        Args:
            symbol: Market token identifier.

        Returns:
            A ``QuoteResult`` containing the best bid, best ask, and calculated mid price.
        """
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0].get("price", 0)) if bids else 0.0
        best_ask = float(asks[0].get("price", 1)) if asks else 1.0
        mid = (best_bid + best_ask) / 2
        return QuoteResult(symbol=symbol, bid=best_bid, ask=best_ask, last=mid)

    async def get_historical(
        self, symbol: str, interval: str = "1d", limit: int = 500
    ) -> List[Dict]:
        """Polymarket doesn't have traditional OHLCV data.

        Args:
            symbol: Market token identifier.
            interval: Desired time granularity (ignored).
            limit: Maximum number of data points to return (ignored).

        Returns:
            An empty list, as historical price data is not provided by this client.
        """
        return []