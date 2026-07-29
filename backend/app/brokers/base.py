from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict


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
# Unit tests for boundary conditions
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import unittest
    import asyncio

    class DummyBroker(AbstractBroker):
        """A minimal concrete broker for testing abstract method contracts."""

        async def place_order(self, request: OrderRequest) -> OrderResult:
            return OrderResult(broker_order_id="dummy123", status="filled", filled_qty=request.quantity)

        async def cancel_order(self, broker_order_id: str) -> bool:
            return broker_order_id == "dummy123"

        async def get_order(self, broker_order_id: str) -> dict:
            return {"id": broker_order_id, "status": "filled"}

        async def get_positions(self) -> list[dict]:
            return [{"symbol": "TEST", "qty": 1}]

        async def get_account(self) -> dict:
            return {"balance": 10000, "equity": 10000}

        async def get_quote(self, symbol: str) -> QuoteResult:
            return QuoteResult(symbol=symbol, bid=1.0, ask=1.1, last=1.05)

        async def get_historical(
            self, symbol: str, interval: str, limit: int = 500
        ) -> list[dict]:
            # Return exactly 'limit' dummy bars; if limit <= 0 return empty list.
            if limit <= 0:
                return []
            return [
                {"ts": i, "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 1000}
                for i in range(limit)
            ]

    class TestBaseBroker(unittest.IsolatedAsyncioTestCase):
        async def asyncSetUp(self):
            self.broker = DummyBroker()

        async def test_order_request_zero_quantity(self):
            """Edge case: quantity set to zero should still instantiate."""
            req = OrderRequest(
                symbol="ZERO",
                side="buy",
                order_type="market",
                quantity=0.0,
                limit_price=None,
                stop_price=None,
                time_in_force="GTC",
                account_id="acc1"
            )
            self.assertEqual(req.quantity, 0.0)
            result = await self.broker.place_order(req)
            self.assertEqual(result.filled_qty, 0.0)
            self.assertEqual(result.status, "filled")

        async def test_order_result_defaults(self):
            """Ensure default fields are correctly set when not provided."""
            res = OrderResult(broker_order_id="id123", status="pending")
            self.assertEqual(res.filled_qty, 0.0)
            self.assertIsNone(res.avg_fill_price)
            self.assertIsNone(res.raw_payload)

        async def test_get_historical_boundary_limit(self):
            """Boundary condition: limit <= 0 should return an empty list."""
            empty = await self.broker.get_historical(symbol="TEST", interval="1m", limit=0)
            self.assertIsInstance(empty, list)
            self.assertEqual(len(empty), 0)

            negative = await self.broker.get_historical(symbol="TEST", interval="1m", limit=-5)
            self.assertIsInstance(negative, list)
            self.assertEqual(len(negative), 0)

            # Normal case: limit=3 returns exactly three items
            three = await self.broker.get_historical(symbol="TEST", interval="1m", limit=3)
            self.assertEqual(len(three), 3)
            self.assertTrue(all(isinstance(item, dict) for item in three))

    unittest.main()