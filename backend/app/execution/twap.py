"""
TWAP (Time-Weighted Average Price) execution.

Splits large orders into N equal slices over a given duration.
Minimizes market impact for large positions while providing robust error
handling and structured logging.
"""

import asyncio
from dataclasses import asdict

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult

# Import a specific broker error if it exists; otherwise fall back to a generic base.
try:
    from app.brokers.base import BrokerError  # type: ignore
except ImportError:  # pragma: no cover
    class BrokerError(Exception):
        """Fallback generic broker error when a specific class is unavailable."""
        pass

from app.utils.logging import logger


class TWAPExecution:
    """
    Execute an order using the Time‑Weighted Average Price (TWAP) strategy.

    Parameters
    ----------
    broker: AbstractBroker
        The broker implementation used to place individual slice orders.
    slices: int, default 10
        Number of equal slices the total quantity will be divided into.
    duration_minutes: int, default 30
        Total duration over which the slices are executed.
    """

    def __init__(self, broker: AbstractBroker, slices: int = 10, duration_minutes: int = 30):
        self.broker = broker
        self.slices = slices
        self.sleep_seconds = (duration_minutes * 60) / slices

    async def execute(self, request: OrderRequest) -> OrderResult:
        """
        Execute the given order request using the TWAP algorithm.

        The method attempts to place ``self.slices`` market orders of equal
        size.  If three consecutive slice placements fail, execution aborts.

        Parameters
        ----------
        request: OrderRequest
            The original order request to be split.

        Returns
        -------
        OrderResult
            Aggregated result covering all slices that were placed.
        """
        slice_qty = request.quantity / self.slices
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None
        consecutive_failures = 0

        for i in range(self.slices):
            slice_req = OrderRequest(
                **{**asdict(request), "quantity": slice_qty, "order_type": "market"}
            )
            try:
                result = await self.broker.place_order(slice_req)
                total_filled += result.filled_qty
                if result.avg_fill_price:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result
                consecutive_failures = 0
            except BrokerError as e:
                consecutive_failures += 1
                logger.error(
                    "TWAP slice broker error",
                    extra={
                        "symbol": request.symbol,
                        "slice_index": i + 1,
                        "total_slices": self.slices,
                        "error": str(e),
                    },
                    exc_info=True,
                )
                if consecutive_failures >= 3:
                    logger.error(
                        "TWAP aborting after consecutive broker failures",
                        extra={
                            "symbol": request.symbol,
                            "failed_slices": consecutive_failures,
                        },
                    )
                    break
            except Exception as e:  # pragma: no cover
                # Catch any unexpected exception types while still providing
                # structured logging.
                consecutive_failures += 1
                logger.exception(
                    "TWAP slice unexpected error",
                    extra={
                        "symbol": request.symbol,
                        "slice_index": i + 1,
                        "total_slices": self.slices,
                        "error": str(e),
                    },
                )
                if consecutive_failures >= 3:
                    logger.error(
                        "TWAP aborting after consecutive unexpected failures",
                        extra={
                            "symbol": request.symbol,
                            "failed_slices": consecutive_failures,
                        },
                    )
                    break

            if i < self.slices - 1:
                await asyncio.sleep(self.sleep_seconds)

        avg_price = total_cost / total_filled if total_filled > 0 else None
        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "twap",
            status="filled"
            if total_filled >= request.quantity * 0.95
            else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )