"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
"""
import time
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
        self.signal_count = 0  # Tracks number of order signals processed

    async def get_markets(self, min_open_interest: float = 10000) -> list[dict]:
        """Auto-discover active markets with sufficient liquidity."""
        start_time = time.time()
        try:
            import asyncio
            markets = await asyncio.to_thread(self.client.get_markets)
            filtered = [m for m in markets if float(m.get("openInterest", 0)) >= min_open_interest]
            elapsed = time.time() - start_time
            logger.info(
                "Polymarket get_markets completed",
                market_count=len(filtered),
                execution_time=elapsed,
                min_open_interest=min_open_interest,
            )
            return filtered
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "Polymarket market fetch failed",
                error=str(e),
                execution_time=elapsed,
            )
            return []

    async def get_order_book(self, token_id: str) -> dict:
        start_time = time.time()
        import asyncio
        result = await asyncio.to_thread(self.client.get_order_book, token_id)
        elapsed = time.time() - start_time
        logger.info(
            "Polymarket get_order_book completed",
            token_id=token_id,
            execution_time=elapsed,
        )
        return result

    async def place_order(self, request: OrderRequest) -> OrderResult:
        start_time = time.time()
        try:
            import asyncio
            args = OrderArgs(
                token_id=request.symbol,
                price=request.limit_price or 0.5,
                size=request.quantity,
                side=request.side.upper(),
            )
            order = await asyncio.to_thread(self.client.create_and_post_order, args)
            elapsed = time.time() - start_time
            self.signal_count += 1
            # P&L is not directly known at order placement; placeholder None
            logger.info(
                "Polymarket order placed",
                signal_count=self.signal_count,
                execution_time=elapsed,
                broker_order_id=str(order.get("orderID", "")),
                status=order.get("status", "pending"),
                pnl=None,
            )
            return OrderResult(
                broker_order_id=str(order.get("orderID", "")),
                status=order.get("status", "pending"),
                raw_payload=order,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "Polymarket place_order failed",
                error=str(e),
                execution_time=elapsed,
                request_symbol=request.symbol,
            )
            raise BrokerError(f"Polymarket: {e}")

    async def cancel_order(self, broker_order_id: str) -> bool:
        start_time = time.time()
        try:
            import asyncio
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            elapsed = time.time() - start_time
            logger.info(
                "Polymarket cancel_order succeeded",
                order_id=broker_order_id,
                execution_time=elapsed,
            )
            return True
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(
                "Polymarket cancel_order failed",
                order_id=broker_order_id,
                error=str(e),
                execution_time=elapsed,
            )
            return False

    async def get_order(self, broker_order_id: str) -> dict:
        start_time = time.time()
        import asyncio
        result = await asyncio.to_thread(self.client.get_order, broker_order_id)
        elapsed = time.time() - start_time
        logger.info(
            "Polymarket get_order completed",
            order_id=broker_order_id,
            execution_time=elapsed,
        )
        return result

    async def get_positions(self) -> list[dict]:
        # No position tracking for Polymarket; return empty list
        logger.info("Polymarket get_positions called - returning empty list")
        return []

    async def get_account(self) -> dict:
        # Account details are not exposed; return empty dict
        logger.info("Polymarket get_account called - returning empty dict")
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        start_time = time.time()
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        quote = QuoteResult(
            symbol=symbol,
            bid=best_bid,
            ask=best_ask,
            last=(best_bid + best_ask) / 2,
        )
        elapsed = time.time() - start_time
        logger.info(
            "Polymarket get_quote completed",
            symbol=symbol,
            bid=best_bid,
            ask=best_ask,
            execution_time=elapsed,
        )
        return quote

    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[dict]:
        logger.info(
            "Polymarket get_historical called - Polymarket does not provide OHLCV data",
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
        return []  # Polymarket doesn't have traditional OHLCV