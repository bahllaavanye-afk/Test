from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: str               # buy|sell
    order_type: str         # market|limit|stop|bracket
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None
    stop_loss: float | None = None      # for bracket orders
    take_profit: float | None = None    # for bracket orders
    time_in_force: str = "GTC"
    account_id: str = ""
    strategy_id: str | None = None
    risk_bucket: str = "directional"   # for risk manager routing
    execution_algo: str = "limit_first"  # market|limit_first|twap|vwap


@dataclass(slots=True)
class OrderResult:
    broker_order_id: str
    status: str
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    raw_payload: dict | None = None


@dataclass(slots=True)
class QuoteResult:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float | None = None


class AbstractBroker(ABC):
    """Interface that all brokers must implement."""

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order to the broker. Raises BrokerError on failure."""

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True if cancelled."""

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> dict:
        """Get current status of an order."""

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Return all open positions."""

    @abstractmethod
    async def get_account(self) -> dict:
        """Return account balance and equity."""

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return current bid/ask/last for a symbol."""

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> list[dict]:
        """Return OHLCV bars. Each dict: {ts, open, high, low, close, volume}."""


# ----------------------------------------------------------------------
# Unit tests for edge‑case validation
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import unittest
    import asyncio

    class TestBaseBroker(unittest.IsolatedAsyncioTestCase):
        """Test edge cases for the broker base definitions."""

        def test_abstract_cannot_instantiate(self):
            """AbstractBroker should not be directly instantiable."""
            with self.assertRaises(TypeError):
                AbstractBroker()  # type: ignore[arg-type]

        def test_order_request_boundary_quantities(self):
            """OrderRequest should accept zero and negative quantities (no validation)."""
            # Zero quantity
            zero_qty = OrderRequest(
                symbol="TEST",
                side="buy",
                order_type="market",
                quantity=0.0,
            )
            self.assertEqual(zero_qty.quantity, 0.0)

            # Negative quantity
            negative_qty = OrderRequest(
                symbol="TEST",
                side="sell",
                order_type="limit",
                quantity=-5.0,
                limit_price=100.0,
            )
            self.assertEqual(negative_qty.quantity, -5.0)
            self.assertEqual(negative_qty.limit_price, 100.0)

        async def test_minimal_broker_implementation(self):
            """A minimal concrete broker should satisfy the abstract interface."""
            class MinimalBroker(AbstractBroker):
                async def place_order(self, request: OrderRequest) -> OrderResult:
                    return OrderResult(broker_order_id="id123", status="filled")

                async def cancel_order(self, broker_order_id: str) -> bool:
                    return True

                async def get_order(self, broker_order_id: str) -> dict:
                    return {"id": broker_order_id, "status": "open"}

                async def get_positions(self) -> list[dict]:
                    return [{"symbol": "TEST", "qty": 10}]

                async def get_account(self) -> dict:
                    return {"balance": 1000.0, "equity": 1500.0}

                async def get_quote(self, symbol: str) -> QuoteResult:
                    return QuoteResult(symbol=symbol, bid=99.5, ask=100.5, last=100.0)

                async def get_historical(
                    self, symbol: str, interval: str, limit: int = 500
                ) -> list[dict]:
                    return [{"ts": 0, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 1000}]

            broker = MinimalBroker()

            # place_order returns an OrderResult with expected defaults
            req = OrderRequest(symbol="TEST", side="buy", order_type="market", quantity=1.0)
            result = await broker.place_order(req)
            self.assertIsInstance(result, OrderResult)
            self.assertEqual(result.filled_qty, 0.0)  # default value
            self.assertIsNone(result.avg_fill_price)

            # cancel_order returns True
            cancelled = await broker.cancel_order("id123")
            self.assertTrue(cancelled)

            # get_quote returns a QuoteResult with correct types
            quote = await broker.get_quote("TEST")
            self.assertIsInstance(quote, QuoteResult)
            self.assertGreaterEqual(quote.bid, 0)
            self.assertGreaterEqual(quote.ask, quote.bid)

    unittest.main()