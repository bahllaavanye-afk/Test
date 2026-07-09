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


class ExecutionError(RuntimeError):
    """Custom exception for execution‑level failures."""


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
            # ---- Quote retrieval -------------------------------------------------
            try:
                quote = await self.broker.get_quote(request.symbol)
            except (ConnectionError, asyncio.TimeoutError) as exc:
                raise ExecutionError("Failed to retrieve quote") from exc

            ref_price = quote.ask if request.side == "buy" else quote.bid
            offset = ref_price * self.offset_bps / 10_000

            if request.side == "buy":
                limit_price = quote.ask - offset
            else:
                limit_price = quote.bid + offset

            limit_req = OrderRequest(
                **{**asdict(request), "order_type": "limit", "limit_price": round(limit_price, 4)}
            )

            # ---- Place limit order ------------------------------------------------
            try:
                result = await self.broker.place_order(limit_req)
            except (ConnectionError, asyncio.TimeoutError) as exc:
                raise ExecutionError("Failed to place limit order") from exc

            if result.status in ("filled", "partially_filled"):
                return self._log_and_return(result, signal_id, start_ts, request, ref_price)

            # ---- Wait for fill ----------------------------------------------------
            for _ in range(self.fallback_seconds):
                await asyncio.sleep(1)
                try:
                    order_status = await self.broker.get_order(result.broker_order_id)
                except (ConnectionError, asyncio.TimeoutError) as exc:
                    logger.warning(
                        "Failed to poll order status; will continue waiting",
                        extra={"signal_id": signal_id, "error": str(exc)},
                    )
                    continue

                if order_status.get("status") in ("filled", "closed"):
                    result.status = "filled"
                    result.filled_qty = float(
                        order_status.get("filled_qty", request.quantity)
                    )
                    return self._log_and_return(result, signal_id, start_ts, request, ref_price)

            # ---- Cancel limit and fall back to market ----------------------------
            try:
                await self.broker.cancel_order(result.broker_order_id)
            except (ConnectionError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "Failed to cancel limit order; proceeding with market fallback",
                    extra={"signal_id": signal_id, "error": str(exc)},
                )

            market_req = OrderRequest(**{**asdict(request), "order_type": "market", "limit_price": None})
            try:
                market_result = await self.broker.place_order(market_req)
            except (ConnectionError, asyncio.TimeoutError) as exc:
                raise ExecutionError("Failed to place market order after limit fallback") from exc

            return self._log_and_return(market_result, signal_id, start_ts, request, ref_price)

        except ExecutionError as exec_err:
            logger.error(
                "LimitFirstExecution encountered a recoverable error; falling back to market",
                extra={"signal_id": signal_id, "error": str(exec_err)},
            )
            # Fallback to market order on any execution‑level error
            market_req = OrderRequest(**{**asdict(request), "order_type": "market"})
            try:
                market_result = await self.broker.place_order(market_req)
            except Exception as exc:  # pragma: no cover – last‑ditch failure
                logger.exception(
                    "Market fallback also failed",
                    extra={"signal_id": signal_id, "error": str(exc)},
                )
                raise
            return self._log_and_return(market_result, signal_id, start_ts, request, None)

        except Exception as exc:  # pragma: no cover – unexpected failures
            logger.exception(
                "Unexpected error in LimitFirstExecution; attempting market fallback",
                extra={"signal_id": signal_id, "error": str(exc)},
            )
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