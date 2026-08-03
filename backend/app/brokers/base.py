from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pytest


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
# Unit tests for edge‑case behavior of the data classes
# ----------------------------------------------------------------------


def test_order_request_defaults():
    """Verify that optional fields receive their defined defaults."""
    req = OrderRequest(symbol="AAPL", side="buy", order_type="market", quantity=10)
    assert req.time_in_force == "GTC"
    assert req.account_id == ""
    assert req.strategy_id is None
    assert req.risk_bucket == "directional"
    assert req.execution_algo == "limit_first"


def test_order_result_defaults():
    """Check that OrderResult initializes optional fields correctly."""
    res = OrderResult(broker_order_id="ord-123", status="filled")
    assert res.filled_qty == 0.0
    assert res.avg_fill_price is None
    assert res.raw_payload is None


def test_dataclass_slots_prevent_extra_attribute():
    """Slots should raise AttributeError when adding undeclared attributes."""
    req = OrderRequest(
        symbol="BTCUSD",
        side="sell",
        order_type="limit",
        quantity=5,
        limit_price=30000.0,
    )
    with pytest.raises(AttributeError):
        req.unexpected_field = "should fail"


def test_order_request_large_quantity_boundary():
    """Ensure that very large quantity values are stored without overflow."""
    large_qty = 1e9  # one billion units
    req = OrderRequest(symbol="ETHUSD", side="buy", order_type="market", quantity=large_qty)
    assert req.quantity == large_qty
    # Also verify that limit_price can be None for market orders
    assert req.limit_price is None