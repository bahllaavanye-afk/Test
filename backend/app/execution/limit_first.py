"""
Limit‑First Execution module.

Provides an execution strategy that first posts a limit order at the best bid/ask
adjusted by a configurable offset (in basis points). If the limit order does
not fill within a configurable fallback window, the strategy cancels the limit
order and falls back to a market order. This approach typically saves 5‑15 bps
versus immediate market execution.
"""

import asyncio
import logging
import time
from dataclasses import asdict
from typing import Optional

import pytest

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult

logger = logging.getLogger(__name__)


class LimitFirstExecution:
    """
    Execute orders using a *limit‑first* approach.

    The strategy:
    1. Retrieve the current quote for the symbol.
    2. Compute a limit price by applying ``offset_bps`` to the best bid/ask.
    3. Submit a limit order.
    4. If the limit does not fill within ``fallback_seconds``, cancel it and
       submit a market order.

    Attributes
    ----------
    broker : AbstractBroker
        Broker implementation used to fetch quotes, place, query, and cancel
        orders.
    offset_bps : float
        Offset applied to the reference price (in basis points). Positive values
        move the limit price away from the market to increase fill probability.
    fallback_seconds : int
        Number of seconds to wait for the limit order to fill before falling back
        to a market order.
    """

    _signal_counter: int = 0

    def __init__(self, broker: AbstractBroker, offset_bps: float = 5, fallback_seconds: int = 30) -> None:
        """
        Parameters
        ----------
        broker : AbstractBroker
            Broker instance used for order handling.
        offset_bps : float, optional
            Offset in basis points applied to the reference price (default is 5).
        fallback_seconds : int, optional
            Seconds to wait before falling back to a market order (default is 30).
        """
        self.broker = broker
        self.offset_bps = offset_bps
        self.fallback_seconds = fallback_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        """
        Execute an order using the limit‑first strategy with structured logging.

        Parameters
        ----------
        request : OrderRequest
            The original order request containing symbol, side, quantity, etc.

        Returns
        -------
        OrderResult
            The final order result after either a successful limit fill or a
            fallback market execution.

        Logs
        ----
        Emits an INFO log at the start and completion of execution containing
        metrics such as ``signal_id``, ``symbol``, ``side``, ``quantity``,
        ``execution_time_ms``, ``filled_qty``, ``fill_price``, ``status``, and
        ``pnl`` when calculable.
        """
        # Increment signal counter and capture start time
        LimitFirstExecution._signal_counter += 1
        signal_id = LimitFirstExecution._signal_counter
        start_ts = time.perf_counter()

        logger.info(
            "Starting LimitFirstExecution",
            extra={
                "signal_id": signal_id,
                "symbol": request.symbol,
                "side": request.side,
                "quantity": request.quantity,
                "offset_bps": self.offset_bps,
                "fallback_seconds": self.fallback_seconds,
            },
        )

        try:
            # Get current quote
            quote = await self.broker.get_quote(request.symbol)
            ref_price = quote.ask if request.side == "buy" else quote.bid
            offset = ref_price * self.offset_bps / 10_000

            if request.side == "buy":
                limit_price = quote.ask - offset  # post below ask to improve fill
            else:
                limit_price = quote.bid + offset  # post above bid to improve fill

            limit_req = OrderRequest(
                **{**asdict(request), "order_type": "limit", "limit_price": round(limit_price, 4)}
            )
            result = await self.broker.place_order(limit_req)

            if result.status in ("filled", "partially_filled"):
                # Successful limit fill
                return self._log_and_return(result, signal_id, start_ts, request, ref_price)

            # Wait for fill, then fallback to market
            for _ in range(self.fallback_seconds):
                await asyncio.sleep(1)
                order_status = await self.broker.get_order(result.broker_order_id)
                if order_status.get("status") in ("filled", "closed"):
                    result.status = "filled"
                    result.filled_qty = float(
                        order_status.get("filled_qty", request.quantity)
                    )
                    return self._log_and_return(result, signal_id, start_ts, request, ref_price)

            # Cancel limit and submit market
            await self.broker.cancel_order(result.broker_order_id)
            market_req = OrderRequest(**{**asdict(request), "order_type": "market", "limit_price": None})
            market_result = await self.broker.place_order(market_req)
            return self._log_and_return(market_result, signal_id, start_ts, request, ref_price)

        except Exception as exc:
            logger.exception(
                "LimitFirstExecution encountered an error, falling back to market",
                extra={"signal_id": signal_id, "error": str(exc)},
            )
            # If anything fails, fall back to direct market order
            market_req = OrderRequest(**{**asdict(request), "order_type": "market"})
            market_result = await self.broker.place_order(market_req)
            return self._log_and_return(market_result, signal_id, start_ts, request, None)

    def _log_and_return(
        self,
        result: OrderResult,
        signal_id: int,
        start_ts: float,
        request: OrderRequest,
        reference_price: Optional[float],
    ) -> OrderResult:
        """
        Log execution metrics and return the result.

        Parameters
        ----------
        result : OrderResult
            The order result to be logged.
        signal_id : int
            Incremental identifier for the request.
        start_ts : float
            Timestamp captured at the start of execution (perf_counter).
        request : OrderRequest
            Original order request.
        reference_price : float | None
            The price used as a reference for P&L calculation (best bid/ask).

        Returns
        -------
        OrderResult
            The same ``result`` object after logging.
        """
        end_ts = time.perf_counter()
        exec_time_ms = int((end_ts - start_ts) * 1000)

        # Attempt to extract fill price for P&L calculation
        fill_price = getattr(result, "filled_price", None) or getattr(result, "avg_price", None)

        pnl: Optional[float] = None
        if fill_price is not None and reference_price is not None:
            # Simple P&L: (reference - fill) * quantity for buys, opposite for sells
            qty = getattr(result, "filled_qty", request.quantity)
            if request.side == "buy":
                pnl = (reference_price - fill_price) * qty
            else:
                pnl = (fill_price - reference_price) * qty

        logger.info(
            "LimitFirstExecution completed",
            extra={
                "signal_id": signal_id,
                "symbol": request.symbol,
                "side": request.side,
                "quantity": request.quantity,
                "execution_time_ms": exec_time_ms,
                "filled_qty": getattr(result, "filled_qty", None),
                "fill_price": fill_price,
                "status": result.status,
                "pnl": pnl,
            },
        )
        return result


# ==============================
# Unit tests for edge conditions
# ==============================

class _MockQuote:
    def __init__(self, bid: float, ask: float):
        self.bid = bid
        self.ask = ask


class _MockBroker(AbstractBroker):
    """
    Minimal mock broker to simulate the required async methods.
    It records calls for assertions and can be configured to raise errors.
    """

    def __init__(self, *, quote: _MockQuote = None, raise_on_quote: Exception = None):
        self.quote = quote or _MockQuote(bid=100.0, ask=101.0)
        self.raise_on_quote = raise_on_quote
        self.placed_orders = []
        self.canceled_orders = []
        self.order_statuses = {}
        self._next_order_id = 1

    async def get_quote(self, symbol: str):
        if self.raise_on_quote:
            raise self.raise_on_quote
        return self.quote

    async def place_order(self, order: OrderRequest) -> OrderResult:
        order_id = f"order-{self._next_order_id}"
        self._next_order_id += 1
        self.placed_orders.append((order_id, order))
        # By default, limit orders are not filled; market orders are filled instantly.
        status = "filled" if order.order_type == "market" else "new"
        result = OrderResult(
            broker_order_id=order_id,
            status=status,
            filled_qty=order.quantity if status == "filled" else 0.0,
            filled_price=order.limit_price if order.order_type == "limit" else None,
        )
        # Store a stub status for later polling
        self.order_statuses[order_id] = {"status": status, "filled_qty": order.quantity}
        return result

    async def get_order(self, broker_order_id: str):
        # Return the stored status; if not present, assume still open.
        return self.order_statuses.get(broker_order_id, {"status": "open"})

    async def cancel_order(self, broker_order_id: str):
        self.canceled_orders.append(broker_order_id)
        # Simulate cancellation by marking as canceled.
        self.order_statuses[broker_order_id] = {"status": "canceled", "filled_qty": 0.0}


@pytest.mark.asyncio
async def test_fallback_seconds_zero_triggers_immediate_market():
    """
    When ``fallback_seconds`` is set to 0 the limit order should be cancelled
    immediately and a market order placed without any waiting loop.
    """
    broker = _MockBroker()
    exec_strategy = LimitFirstExecution(broker=broker, offset_bps=5, fallback_seconds=0)

    request = OrderRequest(
        symbol="TEST",
        side="buy",
        quantity=10,
        order_type="limit",  # placeholder, will be overridden
        limit_price=None,
    )
    result = await exec_strategy.execute(request)

    # Verify that a market order was placed last
    assert result.status == "filled"
    placed_order_types = [order.order_type for _, order in broker.placed_orders]
    # The first placed order is the limit, the second should be market
    assert placed_order_types == ["limit", "market"]
    # The limit order should have been cancelled
    assert len(broker.canceled_orders) == 1
    assert broker.canceled_orders[0] == broker.placed_orders[0][0]


@pytest.mark.asyncio
async def test_offset_bps_zero_results_in_equal_limit_price():
    """
    With ``offset_bps`` set to 0 the limit price should equal the reference price
    (ask for buys, bid for sells). This test validates the calculation.
    """
    broker = _MockBroker(quote=_MockQuote(bid=99.5, ask=100.5))
    exec_strategy = LimitFirstExecution(broker=broker, offset_bps=0, fallback_seconds=1)

    request = OrderRequest(
        symbol="TEST",
        side="sell",
        quantity=5,
        order_type="limit",
        limit_price=None,
    )
    await exec_strategy.execute(request)

    # The first placed order is the limit order; its limit_price should match the bid.
    limit_order_id, limit_order = broker.placed_orders[0]
    assert limit_order.order_type == "limit"
    assert limit_order.limit_price == round(broker.quote.bid, 4)


@pytest.mark.asyncio
async def test_exception_during_get_quote_falls_back_to_market():
    """
    If ``get_quote`` raises an exception, the strategy should catch it and
    immediately place a market order.
    """
    broker = _MockBroker(raise_on_quote=RuntimeError("quote failure"))
    exec_strategy = LimitFirstExecution(broker=broker, offset_bps=5, fallback_seconds=5)

    request = OrderRequest(
        symbol="TEST",
        side="buy",
        quantity=1,
        order_type="limit",
        limit_price=None,
    )
    result = await exec_strategy.execute(request)

    # Only one order should be placed (the fallback market order)
    assert len(broker.placed_orders) == 1
    order_id, placed_order = broker.placed_orders[0]
    assert placed_order.order_type == "market"
    assert result.status == "filled"
    # No cancellation should have occurred because no limit order existed
    assert broker.canceled_orders == []