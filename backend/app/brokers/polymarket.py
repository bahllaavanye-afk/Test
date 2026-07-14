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

# ---------- Broker Implementation ----------
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
            return [m for m in markets if float(m.get("openInterest", 0)) >= min_open_interest]
        except Exception as e:
            logger.error("Polymarket market fetch failed", error=str(e))
            return []

    async def get_order_book(self, token_id: str) -> dict:
        import asyncio
        return await asyncio.to_thread(self.client.get_order_book, token_id)

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
            return OrderResult(
                broker_order_id=str(order.get("orderID", "")),
                status=order.get("status", "pending"),
                raw_payload=order,
            )
        except Exception as e:
            raise BrokerError(f"Polymarket: {e}")

    async def cancel_order(self, broker_order_id: str) -> bool:
        try:
            import asyncio
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            return True
        except Exception as e:
            logger.warning("Polymarket cancel_order failed", order_id=broker_order_id, error=str(e))
            return False

    async def get_order(self, broker_order_id: str) -> dict:
        import asyncio
        return await asyncio.to_thread(self.client.get_order, broker_order_id)

    async def get_positions(self) -> list[dict]:
        return []

    async def get_account(self) -> dict:
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        return QuoteResult(symbol=symbol, bid=best_bid, ask=best_ask, last=(best_bid + best_ask) / 2)

    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[dict]:
        return []  # Polymarket doesn't have traditional OHLCV


# ---------- Unit Tests ----------
import unittest
from unittest.mock import patch

class TestPolymarketBroker(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Patch the ClobClient used inside PolymarketBroker
        self.patcher = patch('backend.app.brokers.polymarket.ClobClient')
        MockClient = self.patcher.start()
        self.mock_client = MockClient.return_value
        self.broker = PolymarketBroker(private_key='test_key', chain_id=137)

    async def asyncTearDown(self):
        self.patcher.stop()

    async def test_get_quote_empty_order_book(self):
        """Quote should fallback to default bid/ask when order book is empty."""
        self.mock_client.get_order_book.return_value = {}
        quote = await self.broker.get_quote('TOKEN123')
        self.assertEqual(quote.bid, 0.0)
        self.assertEqual(quote.ask, 1.0)
        self.assertEqual(quote.last, 0.5)

    async def test_get_quote_with_prices(self):
        """Quote should reflect the best bid and ask from the order book."""
        self.mock_client.get_order_book.return_value = {
            "bids": [{"price": "0.2"}],
            "asks": [{"price": "0.8"}],
        }
        quote = await self.broker.get_quote('TOKEN123')
        self.assertAlmostEqual(quote.bid, 0.2)
        self.assertAlmostEqual(quote.ask, 0.8)
        self.assertAlmostEqual(quote.last, 0.5)

    async def test_place_order_uses_default_price_when_none(self):
        """When limit_price is omitted, the broker should use the default price of 0.5."""
        self.mock_client.create_and_post_order.return_value = {"orderID": "order-456", "status": "filled"}
        request = OrderRequest(symbol='TOKEN123', quantity=10, side='buy')
        result = await self.broker.place_order(request)
        self.assertEqual(result.broker_order_id, "order-456")
        self.assertEqual(result.status, "filled")
        # Verify that the OrderArgs passed to the client used the default price
        args_passed = self.mock_client.create_and_post_order.call_args[0][0]
        self.assertEqual(args_passed.price, 0.5)

if __name__ == "__main__":
    unittest.main()