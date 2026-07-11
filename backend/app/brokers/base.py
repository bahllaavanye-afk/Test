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


# ---------------------------------------------------------------------------
# Unit tests for edge‑case validation of the data classes defined above.
# ---------------------------------------------------------------------------
import unittest


class TestBrokerBaseDataclasses(unittest.TestCase):
    def test_order_request_slots_enforce_attribute_error(self):
        """Attempting to set an undefined attribute should raise AttributeError."""
        req = OrderRequest(
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=10.0,
        )
        with self.assertRaises(AttributeError):
            req.undefined_attribute = "should fail"

    def test_order_result_default_values(self):
        """Default fields of OrderResult must match the specification."""
        result = OrderResult(broker_order_id="12345", status="filled")
        self.assertEqual(result.filled_qty, 0.0)
        self.assertIsNone(result.avg_fill_price)
        self.assertIsNone(result.raw_payload)

    def test_quote_result_volume_optional(self):
        """QuoteResult should allow volume to be None without error."""
        quote = QuoteResult(symbol="MSFT", bid=250.0, ask=251.0, last=250.5)
        self.assertIsNone(quote.volume)
        # Explicitly set volume to a numeric value
        quote_with_vol = QuoteResult(symbol="MSFT", bid=250.0, ask=251.0, last=250.5, volume=1_000_000)
        self.assertEqual(quote_with_vol.volume, 1_000_000)


if __name__ == "__main__":
    unittest.main()