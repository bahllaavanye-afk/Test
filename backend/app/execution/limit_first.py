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
from dataclasses import asdict, dataclass
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


# --------------------------------------------------------------------------- #
# Unit tests for edge‑case behavior
# --------------------------------------------------------------------------- #

import unittest

@dataclass
class Quote:
    bid: float
    ask: float


class MockBroker(AbstractBroker):
    """
    Minimal mock broker used for unit testing the LimitFirstExecution strategy.
    It records placed orders and can be configured to raise exceptions or
    control quote data.
    """

    def __init__(self):
        self.placed_orders = []
        self.quote = Quote(bid=100.0, ask=101.0)
        self.raise_on_get_quote = False

    async def get_quote(self, symbol: str) -> Quote:
        if self.raise_on_get_quote:
            raise RuntimeError("quote retrieval failed")
        return self.quote

    async def place_order(self, order: OrderRequest) -> OrderResult:
        """
        Simulate an immediate fill for any order. The returned OrderResult
        mimics the attributes used by the execution logic.
        """
        # Record the order for later inspection
        self.placed_orders.append(order)

        # Build a lightweight result object
        class SimpleResult:
            def __init__(self, status, broker_order_id, filled_qty, filled_price):
                self.status = status
                self.broker_order_id = broker_order_id
                self.filled_qty = filled_qty
                self.filled_price = filled_price
                # ``avg_price`` is sometimes accessed; keep it in sync
                self.avg_price = filled_price

        # Immediate fill simulation
        filled_price = order.limit_price if order.order_type == "limit" else self.quote.ask
        return SimpleResult(
            status="filled",
            broker_order_id=len(self.placed_orders) - 1,
            filled_qty=order.quantity,
            filled_price=filled_price,
        )

    async def get_order(self, broker_order_id: int) -> dict:
        """
        Return a filled status for the given order ID.
        """
        return {"status": "filled", "filled_qty": self.placed_orders[broker_order_id].quantity}

    async def cancel_order(self, broker_order_id: int) -> None:
        """No‑op for the mock."""
        return None


class TestLimitFirstExecution(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.broker = MockBroker()
        self.request = OrderRequest(symbol="TEST", side="buy", quantity=10)

    async def test_zero_fallback_seconds_immediate_market_fallback(self):
        """When fallback_seconds is 0 the limit order should be cancelled immediately
        and a market order placed."""
        exec = LimitFirstExecution(self.broker, offset_bps=5, fallback_seconds=0)
        result = await exec.execute(self.request)

        # The last placed order must be a market order
        self.assertEqual(self.broker.placed_orders[-1].order_type, "market")
        self.assertEqual(result.status, "filled")

    async def test_zero_offset_bps_limit_price_equals_reference(self):
        """With offset_bps set to 0 the limit price should match the reference price."""
        exec = LimitFirstExecution(self.broker, offset_bps=0, fallback_seconds=1)
        # Ensure a known quote
        self.broker.quote = Quote(bid=100.0, ask=101.0)

        result = await exec.execute(self.request)

        limit_order = self.broker.placed_orders[0]
        # For a buy, limit price = ask - 0 = ask
        self.assertAlmostEqual(limit_order.limit_price, 101.0)
        self.assertEqual(result.status, "filled")

    async def test_exception_in_get_quote_falls_back_to_market(self):
        """If retrieving the quote raises, execution should fall back to a market order."""
        self.broker.raise_on_get_quote = True
        exec = LimitFirstExecution(self.broker, offset_bps=5, fallback_seconds=30)

        result = await exec.execute(self.request)

        # The final order should be a market order
        self.assertEqual(self.broker.placed_orders[-1].order_type, "market")
        self.assertEqual(result.status, "filled")


if __name__ == "__main__":
    unittest.main()