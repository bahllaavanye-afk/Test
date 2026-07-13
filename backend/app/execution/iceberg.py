"""
Iceberg execution: show only a small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding the true size.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger


class IcebergExecution:
    """Handles iceberg order execution by submitting the order in small visible slices.

    The execution logic repeatedly places market orders for a configurable
    percentage of the original quantity until the full amount is filled or an
    error occurs.  After each successful slice a configurable delay is applied
    before the next slice is submitted.
    """

    def __init__(self, broker: AbstractBroker, visible_pct: float = 0.10, refill_delay_seconds: int = 5):
        """
        Args:
            broker: An instance of ``AbstractBroker`` used to place slice orders.
            visible_pct: Fraction of the total order that is visible in each slice.
                Must be between 0 and 1; defaults to 10 %.
            refill_delay_seconds: Seconds to wait between slices when more quantity
                remains to be filled.
        """
        self.broker = broker
        self.visible_pct = visible_pct
        self.refill_delay_seconds = refill_delay_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        """Execute an iceberg order based on the supplied ``OrderRequest``.

        The method breaks the original request into smaller market orders, tracks
        filled quantity and cost, and returns a consolidated ``OrderResult``.

        Args:
            request: The original order details.

        Returns:
            An ``OrderResult`` representing the aggregate outcome of all slices.
        """
        visible_qty = self._calculate_visible_qty(request)
        remaining = request.quantity
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None

        while remaining > 0.01:
            slice_qty = min(visible_qty, remaining)
            slice_req = self._create_slice_request(request, slice_qty)

            try:
                result = await self.broker.place_order(slice_req)
                total_filled, total_cost, remaining, last_result = self._process_slice_result(
                    result, total_filled, total_cost, remaining, last_result
                )
                logger.debug(
                    "Iceberg slice",
                    filled=result.filled_qty,
                    remaining=remaining,
                )
                await self._sleep_if_needed(remaining)
            except Exception as e:  # pragma: no cover
                logger.warning("Iceberg slice failed", error=str(e))
                break

        avg_price = total_cost / total_filled if total_filled > 0 else None
        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "iceberg",
            status="filled"
            if total_filled >= request.quantity * 0.95
            else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )

    def _calculate_visible_qty(self, request: OrderRequest) -> float:
        """Determine the visible quantity for each slice."""
        return max(1.0, request.quantity * self.visible_pct)

    def _create_slice_request(self, original: OrderRequest, slice_qty: float) -> OrderRequest:
        """Create a new ``OrderRequest`` for a slice of the original order."""
        slice_data = {
            **asdict(original),
            "quantity": slice_qty,
            "order_type": "market",
        }
        return OrderRequest(**slice_data)

    def _process_slice_result(
        self,
        result: OrderResult,
        total_filled: float,
        total_cost: float,
        remaining: float,
        last_result: OrderResult | None,
    ) -> tuple[float, float, float, OrderResult | None]:
        """Update aggregation counters based on a slice result."""
        total_filled += result.filled_qty
        remaining -= result.filled_qty
        if result.avg_fill_price is not None:
            total_cost += result.avg_fill_price * result.filled_qty
        last_result = result
        return total_filled, total_cost, remaining, last_result

    async def _sleep_if_needed(self, remaining: float) -> None:
        """Sleep between slices when there is still quantity left to fill."""
        if remaining > 0.01:
            await asyncio.sleep(self.refill_delay_seconds)