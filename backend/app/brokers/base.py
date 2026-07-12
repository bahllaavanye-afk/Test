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


# -------------------------------------------------------------------------
# Unit tests for edge‑case behavior of the data classes defined above.
# -------------------------------------------------------------------------
import unittest

class TestBrokerDataClasses(unittest.TestCase):
    def test_order_request_defaults_and_boundary_quantity(self):
        """Create an OrderRequest with minimal required fields and a zero quantity."""
        req = OrderRequest(
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=0.0  # boundary condition: zero quantity
        )
        # Verify defaults are applied correctly
        self.assertEqual(req.time_in_force, "GTC")
        self.assertEqual(req.execution_algo, "limit_first")
        self.assertIsNone(req.limit_price)
        self.assertIsNone(req.stop_price)
        self.assertIsNone(req.stop_loss)
        self.assertIsNone(req.take_profit)
        self.assertEqual(req.quantity, 0.0)

        # Slots should prevent adding new attributes
        with self.assertRaises(AttributeError):
            req.new_attribute = "should fail"

    def test_order_result_defaults_and_type_consistency(self):
        """Validate default values and type consistency for OrderResult."""
        result = OrderResult(broker_order_id="12345", status="filled")
        self.assertEqual(result.filled_qty, 0.0)
        self.assertIsNone(result.avg_fill_price)
        self.assertIsNone(result.raw_payload)

        # Ensure that modifying a mutable default (if any) does not affect other instances
        result2 = OrderResult(broker_order_id="67890", status="pending")
        self.assertIsNone(result2.raw_payload)
        self.assertNotEqual(result.broker_order_id, result2.broker_order_id)

    def test_quote_result_optional_volume(self):
        """QuoteResult should accept None for optional volume field."""
        quote = QuoteResult(symbol="MSFT", bid=250.5, ask=251.0, last=250.75, volume=None)
        self.assertIsNone(quote.volume)
        self.assertEqual(quote.symbol, "MSFT")
        self.assertGreater(quote.ask, quote.bid)

        # Slots enforcement: adding attributes raises AttributeError
        with self.assertRaises(AttributeError):
            quote.extra = 42

if __name__ == "__main__":
    unittest.main()