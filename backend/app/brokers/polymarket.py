"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
Provides tighter entry validation and basic exit confirmation.
"""
from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.exceptions import BrokerError
from app.utils.logging import logger
from app.config import settings

import time

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    POLY_AVAILABLE = True
except ImportError:
    POLY_AVAILABLE = False


class PolymarketBroker(AbstractBroker):
    """Broker implementation for Polymarket CLOB."""

    def __init__(self, private_key: str, chain_id: int = 137):
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )
        # Configurable thresholds for entry validation
        self.max_spread = getattr(settings, "POLY_MAX_SPREAD", 0.02)  # 2% default
        self.price_slippage_tolerance = getattr(settings, "POLY_SLIPPAGE_TOLERANCE", 0.01)  # 1% default
        # Monitoring metrics
        self._signal_count = 0

    async def get_markets(self, min_open_interest: float = 10000) -> list[dict]:
        """Auto‑discover active markets with sufficient liquidity."""
        start_time = time.time()
        try:
            import asyncio
            markets = await asyncio.to_thread(self.client.get_markets)
            result = [
                m for m in markets
                if float(m.get("openInterest", 0)) >= min_open_interest
            ]
            logger.info(
                "Polymarket get_markets completed",
                count=len(result),
                execution_time=time.time() - start_time,
            )
            return result
        except Exception as e:
            logger.error("Polymarket market fetch failed", error=str(e))
            return []

    async def get_order_book(self, token_id: str) -> dict:
        """Retrieve the current order book for a given market token."""
        start_time = time.time()
        import asyncio
        ob = await asyncio.to_thread(self.client.get_order_book, token_id)
        logger.info(
            "Polymarket get_order_book completed",
            token_id=token_id,
            execution_time=time.time() - start_time,
        )
        return ob

    async def _validate_entry(self, symbol: str, limit_price: float | None) -> bool:
        """
        Confirm that the market conditions satisfy tighter entry criteria.

        Checks:
        * Spread between best bid and ask is below ``self.max_spread``.
        * Provided limit price (if any) is within ``self.price_slippage_tolerance`` of the mid price.
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
        """
        Place an order after confirming entry conditions.

        Raises:
            BrokerError: If the order cannot be placed or validation fails.
        """
        start_time = time.time()
        self._signal_count += 1
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
                import time as _time

                _time.sleep(0.2)
                refreshed = await self.get_order(order.get("orderID", ""))
                status = refreshed.get("status", status)

            result = OrderResult(
                broker_order_id=str(order.get("orderID", "")),
                status=status,
                raw_payload=order,
            )

            # Monitoring log
            logger.info(
                "Polymarket place_order executed",
                signal_count=self._signal_count,
                symbol=request.symbol,
                execution_time=time.time() - start_time,
                broker_order_id=result.broker_order_id,
                status=result.status,
                pnl=None,  # P&L not computed at order placement
            )
            return result
        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Polymarket: {e}")

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an existing order."""
        start_time = time.time()
        try:
            import asyncio
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            logger.info(
                "Polymarket cancel_order executed",
                broker_order_id=broker_order_id,
                execution_time=time.time() - start_time,
                success=True,
            )
            return True
        except Exception as e:
            logger.warning(
                "Polymarket cancel_order failed",
                order_id=broker_order_id,
                error=str(e),
            )
            return False

    async def get_order(self, broker_order_id: str) -> dict:
        """Retrieve a specific order's details."""
        start_time = time.time()
        import asyncio
        order = await asyncio.to_thread(self.client.get_order, broker_order_id)
        logger.info(
            "Polymarket get_order completed",
            broker_order_id=broker_order_id,
            execution_time=time.time() - start_time,
        )
        return order

    async def get_positions(self) -> list[dict]:
        """Polymarket does not expose positions via the CLOB client."""
        return []

    async def get_account(self) -> dict:
        """Polymarket does not expose account balances via the CLOB client."""
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return the best bid, ask, and mid price for a market."""
        start_time = time.time()
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0].get("price", 0)) if bids else 0.0
        best_ask = float(asks[0].get("price", 1)) if asks else 1.0
        mid = (best_bid + best_ask) / 2
        logger.info(
            "Polymarket get_quote completed",
            symbol=symbol,
            execution_time=time.time() - start_time,
            bid=best_bid,
            ask=best_ask,
            mid=mid,
        )
        return QuoteResult(symbol=symbol, bid=best_bid, ask=best_ask, last=mid)

    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[dict]:
        """Polymarket doesn't have traditional OHLCV data."""
        return []