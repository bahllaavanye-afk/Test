"""
Limit‑First Execution module.

Provides an execution strategy that first posts a limit order at the best bid/ask
adjusted by a configurable offset (in basis points). If the limit order does
not fill within a configurable fallback window, or market conditions deteriorate,
the strategy cancels the limit order and falls back to a market order. This
approach typically saves 5‑15 bps versus immediate market execution while
protecting against adverse price moves.
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
    Execute orders using a *limit‑first* approach with tighter entry checks and
    dynamic exit handling.

    The strategy:
    1. Retrieve the current quote for the symbol.
    2. Verify that the spread is within an acceptable range.
    3. Compute a limit price by applying ``offset_bps`` to the best bid/ask.
    4. Submit a limit order.
    5. While waiting, monitor fill status **and** market drift; cancel early
       if the price moves unfavourably.
    6. If the limit does not fill within ``fallback_seconds`` (or is cancelled
       early), fall back to a market order.

    Attributes
    ----------
    broker : AbstractBroker
        Broker implementation used to fetch quotes, place, query, and cancel
        orders.
    offset_bps : float
        Offset applied to the reference price (in basis points). Positive values
        move the limit price away from the market to increase fill probability.
    fallback_seconds : int
        Maximum seconds to wait for the limit order to fill before falling back
        to a market order.
    max_spread_bps : float
        Upper bound on the bid‑ask spread (in bps) for which the limit‑first
        strategy is considered viable. If the spread exceeds this value the
        strategy will immediately use a market order.
    """

    _signal_counter: int = 0

    def __init__(
        self,
        broker: AbstractBroker,
        offset_bps: float = 5.0,
        fallback_seconds: int = 30,
        max_spread_bps: float = 50.0,
    ) -> None:
        """
        Parameters
        ----------
        broker : AbstractBroker
            Broker instance used for order handling.
        offset_bps : float, optional
            Offset in basis points applied to the reference price (default 5).
        fallback_seconds : int, optional
            Seconds to wait before falling back to a market order (default 30).
        max_spread_bps : float, optional
            Maximum acceptable spread in basis points (default 50). Larger spreads
            trigger an immediate market order.
        """
        if offset_bps < 0:
            raise ValueError("offset_bps must be non‑negative")
        if fallback_seconds <= 0:
            raise ValueError("fallback_seconds must be positive")
        if max_spread_bps <= 0:
            raise ValueError("max_spread_bps must be positive")

        self.broker = broker
        self.offset_bps = offset_bps
        self.fallback_seconds = fallback_seconds
        self.max_spread_bps = max_spread_bps

    async def execute(self, request: OrderRequest) -> OrderResult:
        """
        Execute an order using the limit‑first strategy with enhanced entry
        validation and dynamic exit handling.

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
            # 1️⃣ Retrieve current quote and validate spread
            quote = await self.broker.get_quote(request.symbol)
            if not self._is_spread_acceptable(quote):
                logger.info(
                    "Spread exceeds max_spread_bps, using market order immediately",
                    extra={"signal_id": signal_id, "spread_bps": self._calc_spread_bps(quote)},
                )
                market_req = OrderRequest(**{**asdict(request), "order_type": "market", "limit_price": None})
                market_result = await self.broker.place_order(market_req)
                return self._log_and_return(market_result, signal_id, start_ts, request, None)

            # 2️⃣ Compute limit price
            ref_price = quote.ask if request.side == "buy" else quote.bid
            offset = ref_price * self.offset_bps / 10_000

            limit_price = (
                quote.ask - offset if request.side == "buy" else quote.bid + offset
            )
            limit_price = round(limit_price, 4)

            limit_req = OrderRequest(
                **{**asdict(request), "order_type": "limit", "limit_price": limit_price}
            )
            result = await self.broker.place_order(limit_req)

            # 3️⃣ Immediate fill check
            if result.status in ("filled", "partially_filled"):
                return self._log_and_return(result, signal_id, start_ts, request, ref_price)

            # 4️⃣ Wait loop with dynamic cancellation
            for elapsed in range(self.fallback_seconds):
                await asyncio.sleep(1)

                # Refresh order status
                order_status = await self.broker.get_order(result.broker_order_id)
                if order_status.get("status") in ("filled", "closed"):
                    result.status = "filled"
                    result.filled_qty = float(
                        order_status.get("filled_qty", request.quantity)
                    )
                    return self._log_and_return(result, signal_id, start_ts, request, ref_price)

                # Refresh quote to evaluate price drift
                current_quote = await self.broker.get_quote(request.symbol)
                if self._should_cancel_early(request.side, limit_price, current_quote, offset):
                    logger.info(
                        "Price drift detected, cancelling limit order early",
                        extra={"signal_id": signal_id, "elapsed_seconds": elapsed + 1},
                    )
                    await self.broker.cancel_order(result.broker_order_id)
                    break

            # 5️⃣ Fallback to market order
            await self.broker.cancel_order(result.broker_order_id)
            market_req = OrderRequest(**{**asdict(request), "order_type": "market", "limit_price": None})
            market_result = await self.broker.place_order(market_req)
            return self._log_and_return(market_result, signal_id, start_ts, request, ref_price)

        except Exception as exc:
            logger.exception(
                "LimitFirstExecution encountered an error, falling back to market",
                extra={"signal_id": signal_id, "error": str(exc)},
            )
            # Fallback to market on any unexpected error
            market_req = OrderRequest(**{**asdict(request), "order_type": "market", "limit_price": None})
            market_result = await self.broker.place_order(market_req)
            return self._log_and_return(market_result, signal_id, start_ts, request, None)

    def _is_spread_acceptable(self, quote) -> bool:
        """Return True if the bid‑ask spread is within ``max_spread_bps``."""
        spread_bps = self._calc_spread_bps(quote)
        return spread_bps <= self.max_spread_bps

    @staticmethod
    def _calc_spread_bps(quote) -> float:
        """Calculate spread in basis points from a quote object."""
        if quote.bid <= 0:
            return float("inf")
        return ((quote.ask - quote.bid) / quote.bid) * 10_000

    @staticmethod
    def _should_cancel_early(side: str, limit_price: float, quote, offset: float) -> bool:
        """
        Determine whether the limit order should be cancelled early due to
        adverse price movement.

        Parameters
        ----------
        side : str
            ``"buy"`` or ``"sell"``.
        limit_price : float
            The price at which the limit order was placed.
        quote : object
            Latest market quote containing ``bid`` and ``ask``.
        offset : float
            The absolute offset used when constructing the limit price.

        Returns
        -------
        bool
            True if market drift makes the limit unlikely to fill.
        """
        if side == "buy":
            # If the ask has risen more than the original offset, the limit is now too aggressive
            return quote.ask > limit_price + offset
        else:
            # For sell side, if bid has fallen below the limit by more than the offset
            return quote.bid < limit_price - offset

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

        # Extract fill price for P&L calculation
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