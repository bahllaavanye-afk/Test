from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import unittest


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
class TestBrokerDataClasses(unittest.TestCase):
    def test_zero_quantity_allowed(self):
        """Zero quantity should be accepted by the dataclass."""
        req = OrderRequest(
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=0.0,
        )
        self.assertEqual(req.quantity, 0.0)

    def test_negative_quantity_allowed(self):
        """Negative quantity should be accepted (validation is broker‑specific)."""
        req = OrderRequest(
            symbol="TSLA",
            side="sell",
            order_type="limit",
            quantity=-5,
            limit_price=800.0,
        )
        self.assertEqual(req.quantity, -5)

    def test_slots_prevent_dynamic_attributes(self):
        """Dataclasses with slots must reject adding unknown attributes."""
        req = OrderRequest(
            symbol="MSFT",
            side="buy",
            order_type="market",
            quantity=10,
        )
        with self.assertRaises(AttributeError):
            # Attempt to set an attribute that does not exist
            req.unexpected_attribute = "test"

    def test_quote_result_volume_none(self):
        """QuoteResult should correctly handle a None volume."""
        quote = QuoteResult(symbol="ETHUSD", bid=1800.5, ask=1801.0, last=1800.75, volume=None)
        self.assertIsNone(quote.volume)
        self.assertEqual(quote.symbol, "ETHUSD")
        self.assertEqual(quote.bid, 1800.5)

    def test_order_result_defaults(self):
        """OrderResult defaults for optional fields should be set correctly."""
        result = OrderResult(broker_order_id="12345", status="filled")
        self.assertEqual(result.filled_qty, 0.0)
        self.assertIsNone(result.avg_fill_price)
        self.assertIsNone(result.raw_payload)


if __name__ == "__main__":
    unittest.main()