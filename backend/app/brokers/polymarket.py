"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
"""
import time
import functools
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


def _log_execution(metric_name: str):
    """Decorator to log execution time and basic result metrics."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            result = await func(self, *args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            log_data = {
                "metric": metric_name,
                "execution_time_ms": int(elapsed_ms),
            }
            # Basic signal count for iterable results
            if isinstance(result, (list, dict)):
                log_data["signal_count"] = len(result) if isinstance(result, list) else len(result)
            # Include any extra info supplied by the wrapped method
            extra = getattr(result, "_log_extra", None)
            if isinstance(extra, dict):
                log_data.update(extra)
            logger.info("PolymarketBroker metric", **log_data)
            return result
        return wrapper
    return decorator


class PolymarketBroker(AbstractBroker):
    def __init__(self, private_key: str, chain_id: int = 137):
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )

    @_log_execution("get_markets")
    async def get_markets(self, min_open_interest: float = 10000) -> list[dict]:
        """Auto-discover active markets with sufficient liquidity."""
        try:
            import asyncio
            markets = await asyncio.to_thread(self.client.get_markets)
            filtered = [m for m in markets if float(m.get("openInterest", 0)) >= min_open_interest]
            return filtered
        except Exception as e:
            logger.error("Polymarket market fetch failed", error=str(e))
            return []

    @_log_execution("get_order_book")
    async def get_order_book(self, token_id: str) -> dict:
        import asyncio
        return await asyncio.to_thread(self.client.get_order_book, token_id)

    @_log_execution("place_order")
    async def place_order(self, request: OrderRequest) -> OrderResult:
        try:
            import asyncio
            args = OrderArgs(
                token_id=request.symbol,
                price=request.limit_price or 0.5,
                size=request.quantity,
                side=request.side.upper(),
            )
            order = await asyncio.to_thread(self.client.create_and_post_order, args)
            result = OrderResult(
                broker_order_id=str(order.get("orderID", "")),
                status=order.get("status", "pending"),
                raw_payload=order,
            )
            # Attach P&L placeholder (actual P&L calculation occurs elsewhere)
            result._log_extra = {"pnl": 0.0}
            return result
        except Exception as e:
            raise BrokerError(f"Polymarket: {e}")

    @_log_execution("cancel_order")
    async def cancel_order(self, broker_order_id: str) -> bool:
        try:
            import asyncio
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            return True
        except Exception as e:
            logger.warning("Polymarket cancel_order failed", order_id=broker_order_id, error=str(e))
            return False

    @_log_execution("get_order")
    async def get_order(self, broker_order_id: str) -> dict:
        import asyncio
        return await asyncio.to_thread(self.client.get_order, broker_order_id)

    @_log_execution("get_positions")
    async def get_positions(self) -> list[dict]:
        return []

    @_log_execution("get_account")
    async def get_account(self) -> dict:
        return {}

    @_log_execution("get_quote")
    async def get_quote(self, symbol: str) -> QuoteResult:
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
        # Attach quote-specific metrics
        quote._log_extra = {"bid": best_bid, "ask": best_ask}
        return quote

    @_log_execution("get_historical")
    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[dict]:
        return []  # Polymarket doesn't have traditional OHLCV