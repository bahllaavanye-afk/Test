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
# Unit tests for edge cases and boundary conditions
# ----------------------------------------------------------------------
import unittest
import asyncio

class TestOrderRequestDataclass(unittest.TestCase):
    def test_defaults(self):
        """Verify that optional fields default to None or expected values."""
        req = OrderRequest(
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=10.0,
        )
        self.assertIsNone(req.limit_price)
        self.assertIsNone(req.stop_price)
        self.assertIsNone(req.stop_loss)
        self.assertIsNone(req.take_profit)
        self.assertEqual(req.time_in_force, "GTC")
        self.assertEqual(req.account_id, "")
        self.assertIsNone(req.strategy_id)
        self.assertEqual(req.risk_bucket, "directional")
        self.assertEqual(req.execution_algo, "limit_first")

    def test_boundary_quantities(self):
        """Check that the dataclass accepts zero and negative quantities (no validation)."""
        zero_qty = OrderRequest(
            symbol="MSFT",
            side="sell",
            order_type="limit",
            quantity=0.0,
            limit_price=250.0,
        )
        self.assertEqual(zero_qty.quantity, 0.0)

        negative_qty = OrderRequest(
            symbol="TSLA",
            side="buy",
            order_type="stop",
            quantity=-5.5,
            stop_price=700.0,
        )
        self.assertEqual(negative_qty.quantity, -5.5)


class MinimalBroker(AbstractBroker):
    """A minimal concrete implementation used solely for testing the abstract interface."""

    async def place_order(self, request: OrderRequest) -> OrderResult:
        return OrderResult(
            broker_order_id="test123",
            status="filled",
            filled_qty=request.quantity,
            avg_fill_price=request.limit_price or 100.0,
            raw_payload={"request": request},
        )

    async def cancel_order(self, broker_order_id: str) -> bool:
        return broker_order_id == "test123"

    async def get_order(self, broker_order_id: str) -> dict:
        return {"id": broker_order_id, "status": "filled"}

    async def get_positions(self) -> list[dict]:
        return [{"symbol": "AAPL", "qty": 10}]

    async def get_account(self) -> dict:
        return {"equity": 100000.0, "cash": 50000.0}

    async def get_quote(self, symbol: str) -> QuoteResult:
        return QuoteResult(symbol=symbol, bid=100.0, ask=101.0, last=100.5)

    async def get_historical(self, symbol: str, interval: str, limit: int = 500) -> list[dict]:
        # Return exactly `limit` empty bars to test boundary handling
        return [{"ts": i, "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0} for i in range(limit)]


class TestAbstractBrokerImplementation(unittest.IsolatedAsyncioTestCase):
    async def test_place_order_returns_order_result(self):
        broker = MinimalBroker()
        request = OrderRequest(
            symbol="GOOG",
            side="buy",
            order_type="limit",
            quantity=1.0,
            limit_price=1500.0,
        )
        result = await broker.place_order(request)
        self.assertIsInstance(result, OrderResult)
        self.assertEqual(result.broker_order_id, "test123")
        self.assertEqual(result.filled_qty, 1.0)
        self.assertEqual(result.avg_fill_price, 1500.0)

    async def test_get_historical_limit_boundary(self):
        broker = MinimalBroker()
        # Test limit = 0 (should return empty list)
        zero_limit = await broker.get_historical("IBM", "1m", limit=0)
        self.assertEqual(zero_limit, [])

        # Test limit = 1 (single bar)
        one_limit = await broker.get_historical("IBM", "1m", limit=1)
        self.assertEqual(len(one_limit), 1)
        self.assertIn("ts", one_limit[0])

if __name__ == "__main__":
    unittest.main()