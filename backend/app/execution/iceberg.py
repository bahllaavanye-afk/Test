"""
Iceberg execution: show only small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding true size.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger


class IcebergExecution:
    """Execute an order using the iceberg strategy.

    The order is split into smaller visible slices. After each slice is filled,
    the next slice is submitted after a configurable delay until the total
    quantity is exhausted or an error occurs.
    """

    def __init__(self, broker: AbstractBroker, visible_pct: float = 0.10, refill_delay_seconds: int = 5):
        """
        Args:
            broker: Broker implementation used to place orders.
            visible_pct: Fraction of the total order to expose per slice (0‑1).
            refill_delay_seconds: Seconds to wait before submitting the next slice.
        """
        self.broker = broker
        self.visible_pct = visible_pct
        self.refill_delay_seconds = refill_delay_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        """Run the iceberg execution for the given order request.

        The function iteratively sends smaller market orders until the total
        requested quantity is filled or an exception is raised.

        Args:
            request: Original order request containing the total quantity.

        Returns:
            An OrderResult summarising the aggregated fills.
        """
        visible_qty = self._calculate_visible_qty(request.quantity)
        remaining = request.quantity
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None

        while remaining > 0.01:
            slice_qty = min(visible_qty, remaining)
            slice_req = self._build_slice_request(request, slice_qty)

            try:
                result = await self.broker.place_order(slice_req)
                total_filled, total_cost, remaining = self._update_aggregates(
                    result, total_filled, total_cost, remaining
                )
                last_result = result
                logger.debug("Iceberg slice", filled=result.filled_qty, remaining=remaining)

                if remaining > 0.01:
                    await asyncio.sleep(self.refill_delay_seconds)
            except Exception as e:
                logger.warning("Iceberg slice failed", error=str(e))
                break

        avg_price = total_cost / total_filled if total_filled > 0 else None
        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "iceberg",
            status="filled" if total_filled >= request.quantity * 0.95 else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )

    def _calculate_visible_qty(self, total_quantity: float) -> float:
        """Determine the visible quantity for each slice."""
        return max(1.0, total_quantity * self.visible_pct)

    def _build_slice_request(self, original: OrderRequest, slice_qty: float) -> OrderRequest:
        """Create a new OrderRequest for a single iceberg slice."""
        slice_data = {**asdict(original), "quantity": slice_qty, "order_type": "market"}
        return OrderRequest(**slice_data)

    def _update_aggregates(
        self,
        result: OrderResult,
        total_filled: float,
        total_cost: float,
        remaining: float,
    ) -> tuple[float, float, float]:
        """Update cumulative filled quantity, cost, and remaining amount."""
        total_filled += result.filled_qty
        remaining -= result.filled_qty
        if result.avg_fill_price:
            total_cost += result.avg_fill_price * result.filled_qty
        return total_filled, total_cost, remaining