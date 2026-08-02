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


# ==============================
# Unit tests for edge conditions
# ==============================
import unittest

class TestOrderRequestEdge(unittest.TestCase):
    def test_zero_quantity(self):
        """Zero quantity should be accepted by the dataclass."""
        req = OrderRequest(symbol='AAPL', side='buy', order_type='market', quantity=0)
        self.assertEqual(req.quantity, 0)

    def test_negative_quantity(self):
        """Negative quantity is allowed at construction (validation is upstream)."""
        req = OrderRequest(symbol='AAPL', side='sell', order_type='market', quantity=-5)
        self.assertLess(req.quantity, 0)

    def test_limit_order_without_price(self):
        """A limit order can be instantiated without a limit_price; validation occurs later."""
        req = OrderRequest(symbol='AAPL', side='buy', order_type='limit', quantity=10)
        self.assertIsNone(req.limit_price)

class TestQuoteResultEdge(unittest.TestCase):
    def test_missing_volume(self):
        """Volume is optional and should default to None when omitted."""
        qr = QuoteResult(symbol='BTC', bid=50000.0, ask=50100.0, last=50050.0)
        self.assertIsNone(qr.volume)

# Minimal concrete broker for async method testing
class DummyBroker(AbstractBroker):
    async def place_order(self, request: OrderRequest) -> OrderResult:
        return OrderResult(broker_order_id='dummy123', status='filled')

    async def cancel_order(self, broker_order_id: str) -> bool:
        return True

    async def get_order(self, broker_order_id: str) -> dict:
        return {'id': broker_order_id, 'status': 'filled'}

    async def get_positions(self) -> list[dict]:
        return []

    async def get_account(self) -> dict:
        return {'balance': 100000.0, 'equity': 100000.0}

    async def get_quote(self, symbol: str) -> QuoteResult:
        return QuoteResult(symbol=symbol, bid=1.0, ask=1.1, last=1.05)

    async def get_historical(self, symbol: str, interval: str, limit: int = 500) -> list[dict]:
        return []

class TestDummyBrokerAsync(unittest.IsolatedAsyncioTestCase):
    async def test_place_order_returns_order_result(self):
        broker = DummyBroker()
        req = OrderRequest(symbol='AAPL', side='buy', order_type='market', quantity=1)
        result = await broker.place_order(req)
        self.assertIsInstance(result, OrderResult)
        self.assertEqual(result.broker_order_id, 'dummy123')
        self.assertEqual(result.status, 'filled')

    async def test_get_quote_structure(self):
        broker = DummyBroker()
        quote = await broker.get_quote('AAPL')
        self.assertIsInstance(quote, QuoteResult)
        self.assertEqual(quote.symbol, 'AAPL')
        self.assertGreaterEqual(quote.ask, quote.bid)

if __name__ == '__main__':
    unittest.main()