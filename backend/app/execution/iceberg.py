"""
Iceberg execution: show only small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding true size.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger


class IcebergExecution:
    """
    Executes an order using the iceberg strategy.

    The order is split into visible slices that are submitted sequentially.
    After each slice is filled the remaining quantity is refilled after a
    configurable delay.
    """

    def __init__(self, broker: AbstractBroker, visible_pct: float = 0.10, refill_delay_seconds: int = 5):
        self.broker = broker
        self.visible_pct = visible_pct
        self.refill_delay_seconds = refill_delay_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        """
        Run the iceberg execution for ``request`` and return a consolidated ``OrderResult``.
        """
        visible_qty = self._initial_visible_qty(request)
        remaining = request.quantity
        total_filled = 0.0
        total_cost = 0.0
        last_result: Optional[OrderResult] = None

        while self._has_remaining(remaining):
            slice_qty = min(visible_qty, remaining)
            slice_req = self._build_slice_request(request, slice_qty)

            try:
                result = await self.broker.place_order(slice_req)
                total_filled, total_cost, remaining, last_result = self._update_aggregates(
                    result, total_filled, total_cost, remaining, last_result
                )
                logger.debug("Iceberg slice", filled=result.filled_qty, remaining=remaining)

                if self._has_remaining(remaining):
                    await asyncio.sleep(self.refill_delay_seconds)
            except Exception as e:  # pragma: no cover
                logger.warning("Iceberg slice failed", error=str(e))
                break

        avg_price = total_cost / total_filled if total_filled > 0 else None
        return self._build_final_result(request, last_result, total_filled, avg_price)

    def _initial_visible_qty(self, request: OrderRequest) -> float:
        """Calculate the initial visible quantity based on the configured percentage."""
        return max(1.0, request.quantity * self.visible_pct)

    def _has_remaining(self, remaining: float) -> bool:
        """Determine whether the remaining quantity is significant enough to continue."""
        return remaining > 0.01

    def _build_slice_request(self, original: OrderRequest, slice_qty: float) -> OrderRequest:
        """Create a new ``OrderRequest`` for a single iceberg slice."""
        slice_data = {**asdict(original), "quantity": slice_qty, "order_type": "market"}
        return OrderRequest(**slice_data)

    def _update_aggregates(
        self,
        result: OrderResult,
        total_filled: float,
        total_cost: float,
        remaining: float,
        last_result: Optional[OrderResult],
    ) -> tuple[float, float, float, OrderResult]:
        """
        Update running totals and return the new state.

        Returns a tuple of (total_filled, total_cost, remaining, last_result).
        """
        total_filled += result.filled_qty
        remaining -= result.filled_qty
        if result.avg_fill_price is not None:
            total_cost += result.avg_fill_price * result.filled_qty
        return total_filled, total_cost, remaining, result

    def _build_final_result(
        self,
        request: OrderRequest,
        last_result: Optional[OrderResult],
        total_filled: float,
        avg_price: Optional[float],
    ) -> OrderResult:
        """
        Assemble the final ``OrderResult`` from accumulated execution data.
        """
        status = "filled" if total_filled >= request.quantity * 0.95 else "partial"
        broker_order_id = last_result.broker_order_id if last_result else "iceberg"
        return OrderResult(
            broker_order_id=broker_order_id,
            status=status,
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )