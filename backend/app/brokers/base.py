"""Base broker interface and data structures.

This module defines the core data classes used for order placement and result
handling, as well as the abstract base class that concrete broker implementations
must inherit from. The classes include detailed type annotations and docstrings
to aid developers and static analysis tools.
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
        Ticker symbol to trade (e.g., ``"AAPL"``).
    side: str
        Order side, either ``"buy"`` or ``"sell"``.
    order_type: str
        Type of order: ``"market"``, ``"limit"``, ``"stop"``, or ``"bracket"``.
    quantity: float
        Number of units/contracts to trade.
    limit_price: Optional[float]
        Limit price for limit orders; ``None`` for market orders.
    stop_price: Optional[float]
        Stop price for stop orders; ``None`` otherwise.
    stop_loss: Optional[float]
        Stop‑loss price for bracket orders.
    take_profit: Optional[float]
        Take‑profit price for bracket orders.
    time_in_force: str
        Order time‑in‑force policy (default ``"GTC"``).
    account_id: str
        Identifier of the account to use.
    strategy_id: Optional[str]
        Identifier of the originating strategy.
    risk_bucket: str
        Risk bucket label for routing in the risk manager.
    execution_algo: str
        Execution algorithm preference (e.g., ``"limit_first"``,
        ``"market"``, ``"twap"``, ``"vwap"``).
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
    """Result returned after an order is submitted.

    Attributes
    ----------
    broker_order_id: str
        Unique identifier assigned by the broker.
    status: str
        Current order status (e.g., ``"filled"``, ``"rejected"``, ``"pending"``).
    filled_qty: float
        Quantity that has been filled; defaults to ``0.0``.
    avg_fill_price: Optional[float]
        Average price at which the order was filled.
    raw_payload: Optional[dict]
        Raw broker response payload for debugging or audit purposes.
    """

    broker_order_id: str
    status: str
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    raw_payload: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class QuoteResult:
    """Realtime quote information for a symbol.

    Attributes
    ----------
    symbol: str
        Ticker symbol.
    bid: float
        Current best bid price.
    ask: float
        Current best ask price.
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
    """Interface that all broker implementations must conform to."""

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order to the broker.

        Parameters
        ----------
        request: OrderRequest
            The order details to be sent to the broker.

        Returns
        -------
        OrderResult
            The broker's acknowledgment and execution details.

        Raises
        ------
        BrokerError
            If the order cannot be placed due to validation or connectivity issues.
        """

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order.

        Parameters
        ----------
        broker_order_id: str
            The identifier of the order to cancel.

        Returns
        -------
        bool
            ``True`` if the cancellation succeeded; otherwise ``False``.
        """

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """Retrieve the current status of an order.

        Parameters
        ----------
        broker_order_id: str
            The broker-assigned order identifier.

        Returns
        -------
        dict
            A dictionary containing order details as provided by the broker.
        """

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch all open positions for the account.

        Returns
        -------
        list[dict]
            A list of dictionaries, each representing an open position.
        """

    @abstractmethod
    async def get_account(self) -> Dict[str, Any]:
        """Obtain account balance and equity information.

        Returns
        -------
        dict
            Account summary including cash balance, margin, equity, etc.
        """

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """Get the latest market quote for a symbol.

        Parameters
        ----------
        symbol: str
            Ticker symbol to query.

        Returns
        -------
        QuoteResult
            Structured quote data containing bid, ask, last price, and optional volume.
        """

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Retrieve historical OHLCV bars.

        Parameters
        ----------
        symbol: str
            Ticker symbol to retrieve data for.
        interval: str
            Bar interval (e.g., ``"1m"``, ``"5m"``, ``"1h"``, ``"1d"``).
        limit: int, default 500
            Maximum number of bars to return.

        Returns
        -------
        list[dict]
            Each dict contains ``ts``, ``open``, ``high``, ``low``, ``close``, and ``volume``.
        """