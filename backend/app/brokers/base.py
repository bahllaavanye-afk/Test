from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

_ALLOWED_SIDES = {"buy", "sell"}
_ALLOWED_ORDER_TYPES = {"market", "limit", "stop", "bracket"}
_ALLOWED_TIME_IN_FORCE = {"GTC", "IOC", "FOK", "Day"}
_ALLOWED_EXEC_ALGOS = {"market", "limit_first", "twap", "vwap"}


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

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non‑empty string")
        if self.side not in _ALLOWED_SIDES:
            raise ValueError(f"side must be one of {_ALLOWED_SIDES}, got '{self.side}'")
        if self.order_type not in _ALLOWED_ORDER_TYPES:
            raise ValueError(
                f"order_type must be one of {_ALLOWED_ORDER_TYPES}, got '{self.order_type}'"
            )
        if not isinstance(self.quantity, (int, float)) or self.quantity < 0:
            raise ValueError("quantity must be a non‑negative number")
        if self.limit_price is not None:
            if not isinstance(self.limit_price, (int, float)) or self.limit_price <= 0:
                raise ValueError("limit_price must be a positive number when provided")
        if self.stop_price is not None:
            if not isinstance(self.stop_price, (int, float)) or self.stop_price <= 0:
                raise ValueError("stop_price must be a positive number when provided")
        if self.stop_loss is not None:
            if not isinstance(self.stop_loss, (int, float)) or self.stop_loss <= 0:
                raise ValueError("stop_loss must be a positive number when provided")
        if self.take_profit is not None:
            if not isinstance(self.take_profit, (int, float)) or self.take_profit <= 0:
                raise ValueError("take_profit must be a positive number when provided")
        if self.time_in_force not in _ALLOWED_TIME_IN_FORCE:
            raise ValueError(
                f"time_in_force must be one of {_ALLOWED_TIME_IN_FORCE}, got '{self.time_in_force}'"
            )
        if not isinstance(self.account_id, str):
            raise ValueError("account_id must be a string")
        if self.strategy_id is not None and not isinstance(self.strategy_id, str):
            raise ValueError("strategy_id must be a string when provided")
        if not isinstance(self.risk_bucket, str) or not self.risk_bucket:
            raise ValueError("risk_bucket must be a non‑empty string")
        if self.execution_algo not in _ALLOWED_EXEC_ALGOS:
            raise ValueError(
                f"execution_algo must be one of {_ALLOWED_EXEC_ALGOS}, got '{self.execution_algo}'"
            )


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

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non‑empty string")
        for field_name in ("bid", "ask", "last"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{field_name} must be a non‑negative number")
        if self.volume is not None and (not isinstance(self.volume, (int, float)) or self.volume < 0):
            raise ValueError("volume must be a non‑negative number when provided")


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
import unittest
import asyncio


class DummyBroker(AbstractBroker):
    """A minimal concrete broker used solely for unit testing."""

    async def place_order(self, request: OrderRequest) -> OrderResult:
        # Mimic simple fill logic: if limit_price is provided, use it; otherwise default to 1.0
        avg_price = request.limit_price if request.limit_price is not None else 1.0
        return OrderResult(
            broker_order_id="dummy",
            status="filled",
            filled_qty=request.quantity,
            avg_fill_price=avg_price,
            raw_payload={"request": request},
        )

    async def cancel_order(self, broker_order_id: str) -> bool:
        return True

    async def get_order(self, broker_order_id: str) -> dict:
        return {"broker_order_id": broker_order_id, "status": "unknown"}

    async def get_positions(self) -> list[dict]:
        return []

    async def get_account(self) -> dict:
        return {"balance": 0.0, "equity": 0.0}

    async def get_quote(self, symbol: str) -> QuoteResult:
        return QuoteResult(symbol=symbol, bid=1.0, ask=1.1, last=1.05, volume=None)

    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> list[dict]:
        return []


class TestOrderRequestBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_zero_quantity(self):
        """Zero quantity should be accepted and result in zero filled quantity."""
        req = OrderRequest(
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=0.0,
        )
        broker = DummyBroker()
        result = await broker.place_order(req)
        self.assertEqual(result.filled_qty, 0.0)
        self.assertEqual(result.avg_fill_price, 1.0)  # default price when limit_price is None

    async def test_missing_limit_price_on_limit_order(self):
        """A limit order without a limit_price should fall back to the broker's default price."""
        req = OrderRequest(
            symbol="MSFT",
            side="sell",
            order_type="limit",
            quantity=10.0,
            limit_price=None,  # explicitly omitted
        )
        broker = DummyBroker()
        result = await broker.place_order(req)
        self.assertEqual(result.avg_fill_price, 1.0)  # default price used
        self.assertEqual(result.filled_qty, 10.0)

    async def test_extreme_price_values(self):
        """Very large price values should be handled without overflow errors."""
        extreme_price = 1e12  # 1 trillion
        req = OrderRequest(
            symbol="GOOG",
            side="buy",
            order_type="limit",
            quantity=1.0,
            limit_price=extreme_price,
        )
        broker = DummyBroker()
        result = await broker.place_order(req)
        self.assertEqual(result.avg_fill_price, extreme_price)
        self.assertEqual(result.filled_qty, 1.0)


if __name__ == "__main__":
    unittest.main()