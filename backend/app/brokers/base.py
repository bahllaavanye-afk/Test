from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass(slots=True)
class OrderRequest:
    """
    Represents a request to place an order with a broker.

    Attributes
    ----------
    symbol: str
        Ticker symbol of the instrument to trade.
    side: str
        Order side, either ``"buy"`` or ``"sell"``.
    order_type: str
        Type of order, e.g., ``"market"``, ``"limit"``, ``"stop"``, or ``"bracket"``.
    quantity: float
        Number of units to trade.
    limit_price: Optional[float]
        Limit price for limit orders; ``None`` if not applicable.
    stop_price: Optional[float]
        Stop price for stop orders; ``None`` if not applicable.
    stop_loss: Optional[float]
        Stop‑loss price for bracket orders; ``None`` if not applicable.
    take_profit: Optional[float]
        Take‑profit price for bracket orders; ``None`` if not applicable.
    time_in_force: str
        Order time‑in‑force policy, defaults to ``"GTC"`` (good‑til‑canceled).
    account_id: str
        Identifier of the account placing the order.
    strategy_id: Optional[str]
        Identifier of the strategy generating the order.
    risk_bucket: str
        Risk bucket used by the risk manager for routing; defaults to ``"directional"``.
    execution_algo: str
        Execution algorithm preference, e.g., ``"limit_first"``, ``"twap"``, ``"vwap"``, or ``"market"``.
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
    """
    Result returned after an order is submitted to a broker.

    Attributes
    ----------
    broker_order_id: str
        Unique identifier assigned by the broker.
    status: str
        Current status of the order (e.g., ``"filled"``, ``"rejected"``, ``"pending"``).
    filled_qty: float
        Quantity that has been filled; defaults to ``0.0``.
    avg_fill_price: Optional[float]
        Average price at which the order was filled; ``None`` if not filled.
    raw_payload: Optional[dict]
        Original response payload from the broker for debugging or auditing.
    """

    broker_order_id: str
    status: str
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    raw_payload: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class QuoteResult:
    """
    Market quote for a single symbol.

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
        Recent traded volume; may be ``None`` if unavailable.
    """

    symbol: str
    bid: float
    ask: float
    last: float
    volume: Optional[float] = None


class AbstractBroker(ABC):
    """Interface that all broker adapters must implement."""

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Submit an order to the broker.

        Parameters
        ----------
        request: OrderRequest
            The order details to be sent to the broker.

        Returns
        -------
        OrderResult
            Result containing broker-assigned order ID and status.

        Raises
        ------
        BrokerError
            If the broker rejects the order or a communication problem occurs.
        """
        ...

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel an open order.

        Parameters
        ----------
        broker_order_id: str
            Identifier of the order to cancel.

        Returns
        -------
        bool
            ``True`` if the cancellation succeeded, ``False`` otherwise.
        """
        ...

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """
        Retrieve the current status of an order.

        Parameters
        ----------
        broker_order_id: str
            Identifier of the order to query.

        Returns
        -------
        dict
            Raw broker response containing order details.
        """
        ...

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions for the account.

        Returns
        -------
        list[dict]
            A list of position dictionaries as returned by the broker.
        """
        ...

    @abstractmethod
    async def get_account(self) -> Dict[str, Any]:
        """
        Retrieve account balance and equity information.

        Returns
        -------
        dict
            Account summary containing balances, equity, margin, etc.
        """
        ...

    @abstractmethod
    async def get_quote(self, symbol: str) -> QuoteResult:
        """
        Obtain the latest market quote for a given symbol.

        Parameters
        ----------
        symbol: str
            Ticker symbol to query.

        Returns
        -------
        QuoteResult
            Dataclass containing bid, ask, last price, and optional volume.
        """
        ...

    @abstractmethod
    async def get_historical(
        self, symbol: str, interval: str, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical OHLCV bars for a symbol.

        Parameters
        ----------
        symbol: str
            Ticker symbol.
        interval: str
            Timeframe for each bar (e.g., ``"1m"``, ``"5m"``, ``"1h"``).
        limit: int, optional
            Maximum number of bars to return; defaults to ``500``.

        Returns
        -------
        list[dict]
            Each dictionary represents a bar with keys ``ts``, ``open``, ``high``,
            ``low``, ``close``, and ``volume``.
        """
        ...