"""
TWAP (Time-Weighted Average Price) execution module.

This module provides a simple TWAP execution strategy that splits a large
order into a configurable number of equal slices and dispatches them at
regular intervals over a specified duration. The implementation aggregates
fill information from each slice and returns a combined ``OrderResult``.
"""

import asyncio

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger


class TWAPExecution:
    """Execute orders using a Time‑Weighted Average Price (TWAP) strategy.

    The order is divided into ``slices`` equal parts and sent to the broker
    at evenly spaced intervals. Fill information from each slice is
    aggregated to produce a final ``OrderResult``.
    """

    def __init__(self, broker: AbstractBroker, slices: int = 10, duration_minutes: int = 30) -> None:
        """
        Initialise a TWAP execution instance.

        Args:
            broker: Broker implementation used to place orders.
            slices: Number of equal slices to split the order into.
            duration_minutes: Total duration (in minutes) over which the slices
                are executed.
        """
        self.broker = broker
        self.slices = slices
        self.sleep_seconds = (duration_minutes * 60) / slices

    async def execute(self, request: OrderRequest) -> OrderResult:
        """
        Execute an order using the TWAP algorithm.

        The request quantity is divided into ``slices`` equal parts, each of
        which is sent as a market order to the broker. The method tracks fill
        quantities and average fill price across all slices and returns a
        consolidated ``OrderResult``.

        Args:
            request: The original order request containing symbol, quantity,
                and other order parameters.

        Returns:
            An ``OrderResult`` representing the aggregated outcome of the TWAP
            execution. The ``status`` is ``filled`` if at least 95 % of the
            requested quantity was filled; otherwise it is ``partial``.
        """
        slice_qty = request.quantity / self.slices
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None
        consecutive_failures = 0

        for i in range(self.slices):
            slice_req = OrderRequest(
                **{**request.__dict__, "quantity": slice_qty, "order_type": "market"}
            )
            try:
                result = await self.broker.place_order(slice_req)
                total_filled += result.filled_qty
                if result.avg_fill_price:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"TWAP slice {i+1}/{self.slices} failed for {request.symbol}: {e}")
                if consecutive_failures >= 3:
                    logger.error(
                        f"TWAP {request.symbol}: {consecutive_failures} consecutive failures — aborting"
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