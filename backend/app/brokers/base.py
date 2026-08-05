from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

"""Broker abstraction layer.

Defines the data structures and abstract interface that concrete broker
implementations must follow. Includes a minimal in‑memory ``DummyBroker``
used for unit testing of request/response handling.
"""


@dataclass(slots=True)
class OrderRequest:
    """Parameters required to place an order with a broker.

    Attributes
    ----------
    symbol: str
        Ticker symbol of the instrument.
    side: str
        ``"buy"`` or ``"sell"``.
    order_type: str
        One of ``"market"``, ``"limit"``, ``"stop"``, ``"bracket"``.
    quantity: float
        Number of units to transact.
    limit_price: float | None, optional
        Price for limit orders; ``None`` if not applicable.
    stop_price: float | None, optional
        Trigger price for stop orders; ``None`` if not applicable.
    stop_loss: float | None, optional
        Stop‑loss price for bracket orders; ``None`` if not applicable.
    take_profit: float | None, optional
        Take‑profit price for bracket orders; ``None`` if not applicable.
    time_in_force: str, default ``"GTC"``
        Order time‑in‑force policy (e.g., ``"GTC"``, ``"IOC"``).
    account_id: str, default ``""``
        Identifier of the account submitting the order.
    strategy_id: str | None, optional
        Identifier of the originating strategy.
    risk_bucket: str, default ``"directional"``
        Bucket used by the risk manager for routing.
    execution_algo: str, default ``"limit_first"``
        Execution algorithm preference.
    """  # noqa: E501

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
    """Result returned after an order submission.

    Attributes
    ----------
    broker_order_id: str
        Unique identifier assigned by the broker.
    status: str
        Order status (e.g., ``"filled"``, ``"rejected"``).
    filled_qty: float, default ``0.0``
        Quantity actually filled.
    avg_fill_price: float | None, optional
        Average price at which the fill occurred.
    raw_payload: dict | None, optional
        Raw broker response payload for debugging/audit purposes.
    """

    broker_order_id: str
    status: str
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    raw_payload: dict | None = None


@dataclass(slots=True)
class QuoteResult:
    """Current market quote for a symbol.

    Attributes
    ----------
    symbol: str
        Ticker symbol.
    bid: float
        Highest bid price.
    ask: float
        Lowest ask price.
    last: float
        Last traded price.
    volume: float | None, optional
        Recent traded volume.
    """

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float | None = None


class AbstractBroker(ABC):
    """Interface that all broker implementations must conform to."""

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order to the broker.

        Parameters
        ----------
        request: OrderRequest
            The order details to be placed.

        Returns
        -------
        OrderResult
            Result containing broker order ID, status, fills, etc.

        Raises
        ------
        BrokerError
            If the order cannot be placed.
        """

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order.

        Parameters
        ----------
        broker_order_id: str
            Identifier of the order to cancel.

        Returns
        -------
        bool
            ``True`` if the order was successfully cancelled.
        """

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """Retrieve the current status of an order.

        Parameters
        ----------
        broker_order_id: str
            Identifier of the order to query.

        Returns
        -------
        dict
            Broker‑specific order details.
        """

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Return all open positions for the account.

        Returns
        -------
        list[dict]
            Each dict represents a position.
        """

    @abstractmethod
    async def get_account(self) -> Dict[str, Any]:
        """Return account balance and equity information.

        Returns
        -------
        dict
            Account summary fields such as ``balance`` and ``equity``.
        """

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Retrieve the latest market quote for a symbol.

        Parameters
        ----------
        symbol: str
            Ticker symbol to quote.

        Returns
        -------
        QuoteResult
            Current bid, ask, last price, and optional volume.
        """

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Fetch historical OHLCV bars.

        Parameters
        ----------
        symbol: str
            Ticker symbol.
        interval: str
            Bar interval (e.g., ``"1m"``, ``"1h"``).
        limit: int, default ``500``
            Maximum number of bars to return.

        Returns
        -------
        list[dict]
            Each dict contains ``ts``, ``open``, ``high``, ``low``, ``close``,
            and ``volume`` keys.
        """


# ----------------------------------------------------------------------
# Unit tests for boundary conditions
# ----------------------------------------------------------------------
import unittest
import asyncio


class DummyBroker(AbstractBroker):
    """A minimal concrete broker used solely for unit testing."""

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Mimic simple fill logic for testing.

        If ``limit_price`` is provided, it is used as the fill price;
        otherwise a default price of ``1.0`` is applied.
        """
        avg_price = request.limit_price if request.limit_price is not None else 1.0
        return OrderResult(
            broker_order_id="dummy",
            status="filled",
            filled_qty=request.quantity,
            avg_fill_price=avg_price,
            raw_payload={"request": request},
        )

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Always succeed in cancelling for test purposes."""
        return True

    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """Return a placeholder order status."""
        return {"broker_order_id": broker_order_id, "status": "unknown"}

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Return an empty position list."""
        return []

    async def get_account(self) -> Dict[str, Any]:
        """Return a zeroed account snapshot."""
        return {"balance": 0.0, "equity": 0.0}

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Provide a static quote for testing."""
        return QuoteResult(symbol=symbol, bid=1.0, ask=1.1, last=1.05, volume=None)

    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Return an empty historical dataset."""
        return []


class TestOrderRequestBoundary(unittest.IsolatedAsyncioTestCase):
    """Boundary condition tests for :class:`OrderRequest` handling."""

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