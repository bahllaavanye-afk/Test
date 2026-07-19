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
            reference_price = await self._get_reference_price(request)
            limit_price = self._compute_limit_price(request, reference_price)

            limit_req = OrderRequest(
                **{**asdict(request), "order_type": "limit", "limit_price": round(limit_price, 4)}
            )
            limit_result = await self.broker.place_order(limit_req)

            if limit_result.status in ("filled", "partially_filled"):
                return self._log_and_return(limit_result, signal_id, start_ts, request, reference_price)

            filled_result = await self._wait_for_fill(limit_result, request, reference_price, start_ts, signal_id)
            if filled_result:
                return filled_result

            # If still not filled, cancel limit and place market order
            await self.broker.cancel_order(limit_result.broker_order_id)
            market_result = await self._place_market_order(request)
            return self._log_and_return(market_result, signal_id, start_ts, request, reference_price)

        except Exception as exc:
            logger.exception(
                "LimitFirstExecution encountered an error, falling back to market",
                extra={"signal_id": signal_id, "error": str(exc)},
            )
            market_result = await self._place_market_order(request)
            return self._log_and_return(market_result, signal_id, start_ts, request, None)

    async def _get_reference_price(self, request: OrderRequest) -> float:
        """Fetch the current quote and return the appropriate reference price."""
        quote = await self.broker.get_quote(request.symbol)
        return quote.ask if request.side == "buy" else quote.bid

    def _compute_limit_price(self, request: OrderRequest, reference_price: float) -> float:
        """Calculate the limit price based on the offset and side."""
        offset = reference_price * self.offset_bps / 10_000
        if request.side == "buy":
            return reference_price - offset  # post below ask
        return reference_price + offset      # post above bid

    async def _wait_for_fill(
        self,
        limit_result: OrderResult,
        request: OrderRequest,
        reference_price: float,
        start_ts: float,
        signal_id: int,
    ) -> Optional[OrderResult]:
        """
        Poll the limit order status for ``fallback_seconds`` seconds.
        Return a logged ``OrderResult`` if filled, otherwise ``None``.
        """
        for _ in range(self.fallback_seconds):
            await asyncio.sleep(1)
            order_status = await self.broker.get_order(limit_result.broker_order_id)
            if order_status.get("status") in ("filled", "closed"):
                limit_result.status = "filled"
                limit_result.filled_qty = float(
                    order_status.get("filled_qty", request.quantity)
                )
                return self._log_and_return(limit_result, signal_id, start_ts, request, reference_price)
        return None

    async def _place_market_order(self, request: OrderRequest) -> OrderResult:
        """Create and place a market order based on the original request."""
        market_req = OrderRequest(**{**asdict(request), "order_type": "market", "limit_price": None})
        return await self.broker.place_order(market_req)

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