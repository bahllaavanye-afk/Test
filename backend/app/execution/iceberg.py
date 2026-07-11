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
    """Execute an order using an iceberg strategy.

    The order is split into visible slices determined by ``visible_pct``.
    After each slice is filled the next slice is placed after ``refill_delay_seconds``.
    """

    def __init__(
        self,
        broker: AbstractBroker,
        visible_pct: float = 0.10,
        refill_delay_seconds: int = 5,
    ) -> None:
        self.broker = broker
        self.visible_pct = visible_pct
        self.refill_delay_seconds = refill_delay_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        """Run the iceberg execution for ``request``."""
        visible_qty = self._initial_visible_qty(request)
        remaining = request.quantity
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None

        while remaining > 0.01:
            slice_qty = min(visible_qty, remaining)
            slice_req = self._build_slice_request(request, slice_qty)

            result = await self._place_slice(slice_req)
            if result is None:
                break

            total_filled, total_cost, remaining, last_result = self._update_totals(
                result, total_filled, total_cost, remaining, last_result
            )

            if remaining > 0.01:
                await asyncio.sleep(self.refill_delay_seconds)

        avg_price = self._average_price(total_cost, total_filled)
        status = self._final_status(total_filled, request.quantity)

        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "iceberg",
            status=status,
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )

    def _initial_visible_qty(self, request: OrderRequest) -> float:
        """Calculate the initial visible quantity for a slice."""
        return max(1.0, request.quantity * self.visible_pct)

    def _build_slice_request(self, request: OrderRequest, slice_qty: float) -> OrderRequest:
        """Create an ``OrderRequest`` for a single iceberg slice."""
        base_dict = asdict(request)
        base_dict.update({"quantity": slice_qty, "order_type": "market"})
        return OrderRequest(**base_dict)

    async def _place_slice(self, slice_req: OrderRequest) -> OrderResult | None:
        """Place a slice order and handle any exception."""
        try:
            result = await self.broker.place_order(slice_req)
            logger.debug(
                "Iceberg slice",
                filled=result.filled_qty,
                remaining=slice_req.quantity - result.filled_qty,
            )
            return result
        except Exception as e:  # pragma: no cover
            logger.warning("Iceberg slice failed", error=str(e))
            return None

    def _update_totals(
        self,
        result: OrderResult,
        total_filled: float,
        total_cost: float,
        remaining: float,
        last_result: OrderResult | None,
    ) -> tuple[float, float, float, OrderResult]:
        """Update cumulative metrics after a slice completes."""
        total_filled += result.filled_qty
        remaining -= result.filled_qty
        if result.avg_fill_price:
            total_cost += result.avg_fill_price * result.filled_qty
        last_result = result
        return total_filled, total_cost, remaining, last_result

    @staticmethod
    def _average_price(total_cost: float, total_filled: float) -> float | None:
        """Calculate the weighted average fill price."""
        return total_cost / total_filled if total_filled > 0 else None

    @staticmethod
    def _final_status(total_filled: float, requested_qty: float) -> str:
        """Determine final order status based on fill ratio."""
        return "filled" if total_filled >= requested_qty * 0.95 else "partial"