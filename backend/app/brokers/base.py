"""Base definitions for broker interactions.

This module defines data structures and an abstract interface that concrete broker
implementations must follow. The dataclasses represent order requests, results,
and quote information, while :class:`AbstractBroker` specifies the asynchronous
API used throughout the platform.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class OrderRequest:
    """Parameters required to place an order with a broker.

    Attributes
    ----------
    symbol: str
        Ticker symbol to trade.
    side: str
        ``"buy"`` or ``"sell"``.
    order_type: str
        Order type, e.g., ``"market"``, ``"limit"``, ``"stop"``, or ``"bracket"``.
    quantity: float
        Number of units to trade.
    limit_price: Optional[float]
        Limit price for limit orders.
    stop_price: Optional[float]
        Stop price for stop orders.
    stop_loss: Optional[float]
        Stop‑loss price for bracket orders.
    take_profit: Optional[float]
        Take‑profit price for bracket orders.
    time_in_force: str
        Order time‑in‑force, default ``"GTC"`` (good‑til‑canceled).
    account_id: str
        Identifier of the account placing the order.
    strategy_id: Optional[str]
        Identifier of the strategy originating the order.
    risk_bucket: str
        Risk bucket used by the risk manager for routing.
    execution_algo: str
        Execution algorithm, e.g., ``"limit_first"``, ``"twap"``, ``"vwap"``.
    """

    symbol: str
    side: str               # buy|sell
    order_type: str         # market|limit|stop|bracket
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss: Optional[float] = None      # for bracket orders
    take_profit: Optional[float] = None    # for bracket orders
    time_in_force: str = "GTC"
    account_id: str = ""
    strategy_id: Optional[str] = None
    risk_bucket: str = "directional"   # for risk manager routing
    execution_algo: str = "limit_first"  # market|limit_first|twap|vwap


@dataclass(slots=True)
class OrderResult:
    """Result returned after submitting an order.

    Attributes
    ----------
    broker_order_id: str
        Unique identifier assigned by the broker.
    status: str
        Current order status (e.g., ``"filled"``, ``"pending"``, ``"rejected"``).
    filled_qty: float
        Quantity that has been filled.
    avg_fill_price: Optional[float]
        Average price of filled quantity.
    raw_payload: Optional[Dict[str, Any]]
        Original response payload from the broker for debugging/audit.
    """

    broker_order_id: str
    status: str
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    raw_payload: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class QuoteResult:
    """Current market quote for a symbol.

    Attributes
    ----------
    symbol: str
        Ticker symbol.
    bid: float
        Best bid price.
    ask: float
        Best ask price.
    last: float
        Last traded price.
    volume: Optional[float]
        Recent traded volume, if available.
    """

    symbol: str
    bid: float
    ask: float
    last: float
    volume: Optional[float] = None


class AbstractBroker(ABC):
    """Interface that all broker implementations must provide.

    Implementations should be asynchronous and raise a :class:`BrokerError`
    (or subclass) on failure where documented.
    """

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order to the broker.

        Parameters
        ----------
        request: OrderRequest
            The order details to be sent.

        Returns
        -------
        OrderResult
            Result containing broker order ID and status.

        Raises
        ------
        BrokerError
            If the order cannot be placed.
        """

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an existing order.

        Parameters
        ----------
        broker_order_id: str
            Identifier of the order to cancel.

        Returns
        -------
        bool
            ``True`` if the order was successfully cancelled, otherwise ``False``.
        """

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """Retrieve the current status of a specific order.

        Parameters
        ----------
        broker_order_id: str
            Identifier of the order.

        Returns
        -------
        dict
            Broker‑specific order information.
        """

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Return all open positions for the account.

        Returns
        -------
        list[dict]
            Each dict contains position details such as symbol, quantity, and entry price.
        """

    @abstractmethod
    async def get_account(self) -> Dict[str, Any]:
        """Return account balance and equity information.

        Returns
        -------
        dict
            Account metrics like cash balance, margin, and net equity.
        """

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Fetch the latest market quote for a symbol.

        Parameters
        ----------
        symbol: str
            Ticker symbol to query.

        Returns
        -------
        QuoteResult
            Current bid, ask, last price, and optional volume.
        """

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Retrieve historical OHLCV data.

        Parameters
        ----------
        symbol: str
            Ticker symbol.
        interval: str
            Timeframe for each bar (e.g., ``"1m"``, ``"5m"``, ``"1d"``).
        limit: int, optional
            Maximum number of bars to return (default ``500``).

        Returns
        -------
        list[dict]
            Each dict contains ``ts``, ``open``, ``high``, ``low``, ``close``, and ``volume``.
        """