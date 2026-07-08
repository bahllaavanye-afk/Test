"""Base broker definitions and abstract interface.

This module defines data structures for order requests, order results, and quote
information, as well as the ``AbstractBroker`` interface that concrete broker
implementations must follow. The classes are lightweight and use ``slots`` to
minimize memory overhead, which is important in a high‑frequency trading
environment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class OrderRequest:
    """Parameters required to place an order with a broker.

    Attributes:
        symbol: Ticker symbol (e.g., ``"AAPL"``).
        side: Order side – ``"buy"`` or ``"sell"``.
        order_type: Type of order – ``"market"``, ``"limit"``, ``"stop"``,
            or ``"bracket"``.
        quantity: Number of units to trade.
        limit_price: Limit price for ``limit`` or ``bracket`` orders.
        stop_price: Stop price for ``stop`` or ``bracket`` orders.
        stop_loss: Stop‑loss price for ``bracket`` orders.
        take_profit: Take‑profit price for ``bracket`` orders.
        time_in_force: Order time‑in‑force policy (default ``"GTC"``).
        account_id: Identifier of the account to which the order belongs.
        strategy_id: Optional identifier of the originating strategy.
        risk_bucket: Risk‑management bucket name (default ``"directional"``).
        execution_algo: Execution algorithm – ``"market"``, ``"limit_first"``,
            ``"twap"``, or ``"vwap"``.
    """

    symbol: str
    side: str  # buy|sell
    order_type: str  # market|limit|stop|bracket
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss: Optional[float] = None  # for bracket orders
    take_profit: Optional[float] = None  # for bracket orders
    time_in_force: str = "GTC"
    account_id: str = ""
    strategy_id: Optional[str] = None
    risk_bucket: str = "directional"  # for risk manager routing
    execution_algo: str = "limit_first"  # market|limit_first|twap|vwap


@dataclass(slots=True)
class OrderResult:
    """Result returned after an order is submitted.

    Attributes:
        broker_order_id: Unique identifier assigned by the broker.
        status: Current order status (e.g., ``"filled"``, ``"rejected"``).
        filled_qty: Quantity that has been filled so far.
        avg_fill_price: Average price of filled quantity, if any.
        raw_payload: Original response payload from the broker for debugging.
    """

    broker_order_id: str
    status: str
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    raw_payload: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class QuoteResult:
    """Realtime market data snapshot for a symbol.

    Attributes:
        symbol: Ticker symbol.
        bid: Best bid price.
        ask: Best ask price.
        last: Last traded price.
        volume: Optional trading volume for the most recent period.
    """

    symbol: str
    bid: float
    ask: float
    last: float
    volume: Optional[float] = None


class AbstractBroker(ABC):
    """Interface that all broker implementations must satisfy."""

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order to the broker.

        Args:
            request: Fully populated :class:`OrderRequest` instance.

        Returns:
            An :class:`OrderResult` describing the broker's response.

        Raises:
            BrokerError: If the order cannot be placed.
        """

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order.

        Args:
            broker_order_id: Identifier of the order to cancel.

        Returns:
            ``True`` if the order was successfully cancelled, otherwise ``False``.
        """

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """Retrieve the current status of a specific order.

        Args:
            broker_order_id: Identifier of the order to query.

        Returns:
            A dictionary containing order details as provided by the broker.
        """

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Obtain all open positions for the account.

        Returns:
            A list of dictionaries, each representing a position.
        """

    @abstractmethod
    async def get_account(self) -> Dict[str, Any]:
        """Fetch account summary information.

        Returns:
            A dictionary with balance, equity, and other account metrics.
        """

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Retrieve the latest market quote for a symbol.

        Args:
            symbol: Ticker symbol to query.

        Returns:
            A :class:`QuoteResult` containing bid, ask, last price, and volume.
        """

    @abstractmethod
    async def get_historical(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Fetch historical OHLCV bars for a symbol.

        Args:
            symbol: Ticker symbol.
            interval: Timeframe for each bar (e.g., ``"1m"``, ``"5m"``).
            limit: Maximum number of bars to return (default ``500``).

        Returns:
            A list of dictionaries, each with keys ``ts``, ``open``, ``high``,
            ``low``, ``close``, and ``volume``.
        """