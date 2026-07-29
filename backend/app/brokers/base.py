from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import unittest
import asyncio


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
# Unit tests for edge‑case behavior
# ----------------------------------------------------------------------
class TestOrderRequestEdgeCases(unittest.TestCase):
    def test_defaults_and_slots(self):
        # Minimal valid request
        req = OrderRequest(
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=1.0,
        )
        # Verify defaults are set correctly
        self.assertIsNone(req.limit_price)
        self.assertIsNone(req.stop_price)
        self.assertIsNone(req.stop_loss)
        self.assertIsNone(req.take_profit)
        self.assertEqual(req.time_in_force, "GTC")
        self.assertEqual(req.account_id, "")
        self.assertIsNone(req.strategy_id)
        self.assertEqual(req.risk_bucket, "directional")
        self.assertEqual(req.execution_algo, "limit_first")
        # Slots enforce attribute protection
        with self.assertRaises(AttributeError):
            req.new_attribute = 123

    def test_boundary_quantity(self):
        # Zero quantity (edge case) should be allowed by the dataclass itself
        req_zero = OrderRequest(
            symbol="AAPL",
            side="sell",
            order_type="limit",
            quantity=0.0,
            limit_price=150.0,
        )
        self.assertEqual(req_zero.quantity, 0.0)

        # Very large quantity (boundary test)
        huge_qty = 1e12
        req_huge = OrderRequest(
            symbol="BTCUSD",
            side="buy",
            order_type="market",
            quantity=huge_qty,
        )
        self.assertEqual(req_huge.quantity, huge_qty)


class TestAbstractBrokerInstantiation(unittest.TestCase):
    def test_cannot_instantiate_without_all_methods(self):
        # Define a subclass that forgets one abstract method
        class IncompleteBroker(AbstractBroker):
            async def place_order(self, request: OrderRequest) -> OrderResult:
                return OrderResult(broker_order_id="1", status="filled")

            async def cancel_order(self, broker_order_id: str) -> bool:
                return True

            async def get_order(self, broker_order_id: str) -> dict:
                return {}

            async def get_positions(self) -> list[dict]:
                return []

            async def get_account(self) -> dict:
                return {}

            async def get_quote(self, symbol: str) -> QuoteResult:
                return QuoteResult(symbol=symbol, bid=1.0, ask=1.0, last=1.0)

            # Missing get_historical implementation

        with self.assertRaises(TypeError):
            IncompleteBroker()

    def test_dummy_broker_async_methods(self):
        class DummyBroker(AbstractBroker):
            async def place_order(self, request: OrderRequest) -> OrderResult:
                return OrderResult(broker_order_id="dummy", status="accepted")

            async def cancel_order(self, broker_order_id: str) -> bool:
                return True

            async def get_order(self, broker_order_id: str) -> dict:
                return {"id": broker_order_id, "status": "open"}

            async def get_positions(self) -> list[dict]:
                return [{"symbol": "XYZ", "qty": 10}]

            async def get_account(self) -> dict:
                return {"balance": 1000.0, "equity": 1200.0}

            async def get_quote(self, symbol: str) -> QuoteResult:
                return QuoteResult(symbol=symbol, bid=100.0, ask=101.0, last=100.5)

            async def get_historical(
                self, symbol: str, interval: str, limit: int = 500
            ) -> list[dict]:
                return [{"ts": 0, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 1000}][:limit]

        async def run_checks():
            broker = DummyBroker()
            req = OrderRequest(symbol="TEST", side="buy", order_type="market", quantity=1.0)
            result = await broker.place_order(req)
            self.assertIsInstance(result, OrderResult)
            self.assertEqual(result.broker_order_id, "dummy")
            canceled = await broker.cancel_order("dummy")
            self.assertTrue(canceled)
            order_info = await broker.get_order("dummy")
            self.assertIn("status", order_info)
            positions = await broker.get_positions()
            self.assertIsInstance(positions, list)
            account = await broker.get_account()
            self.assertIn("balance", account)
            quote = await broker.get_quote("TEST")
            self.assertIsInstance(quote, QuoteResult)
            hist = await broker.get_historical("TEST", "1m", limit=0)
            self.assertEqual(hist, [])

        asyncio.run(run_checks())


if __name__ == "__main__":
    unittest.main()