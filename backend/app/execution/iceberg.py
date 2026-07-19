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
    the next slice is placed after a configurable delay until the total quantity
    is satisfied or an error occurs.
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
        """Run the iceberg execution for ``request`` and return an aggregated result."""
        visible_qty = self._initial_visible_qty(request)
        remaining = request.quantity
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None

        while remaining > 0.01:
            slice_qty = min(visible_qty, remaining)
            slice_req = self._make_slice_request(request, slice_qty)

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

                if remaining > 0.01:
                    await asyncio.sleep(self.refill_delay_seconds)
            except Exception as e:  # pragma: no cover
                logger.warning("Iceberg slice failed", error=str(e))
                break

        avg_price = self._compute_average_price(total_cost, total_filled)
        return self._build_final_result(last_result, request, total_filled, avg_price)

    def _initial_visible_qty(self, request: OrderRequest) -> float:
        """Calculate the initial visible quantity based on ``visible_pct``."""
        return max(1.0, request.quantity * self.visible_pct)

    def _make_slice_request(self, base: OrderRequest, qty: float) -> OrderRequest:
        """Create a slice ``OrderRequest`` with the given quantity and market order type."""
        slice_data = {**asdict(base), "quantity": qty, "order_type": "market"}
        return OrderRequest(**slice_data)

    def _process_slice_result(
        self,
        result: OrderResult,
        total_filled: float,
        total_cost: float,
        remaining: float,
        last_result: OrderResult | None,
    ) -> tuple[float, float, float, OrderResult]:
        """Update aggregate metrics based on a single slice result."""
        total_filled += result.filled_qty
        remaining -= result.filled_qty
        if result.avg_fill_price:
            total_cost += result.avg_fill_price * result.filled_qty
        last_result = result
        return total_filled, total_cost, remaining, last_result

    def _compute_average_price(self, total_cost: float, total_filled: float) -> float | None:
        """Return the weighted average fill price, or ``None`` if nothing was filled."""
        return total_cost / total_filled if total_filled > 0 else None

    def _build_final_result(
        self,
        last_result: OrderResult | None,
        request: OrderRequest,
        filled_qty: float,
        avg_price: float | None,
    ) -> OrderResult:
        """Construct the final ``OrderResult`` for the whole iceberg execution."""
        broker_order_id = (
            last_result.broker_order_id if last_result else "iceberg"
        )
        status = "filled" if filled_qty >= request.quantity * 0.95 else "partial"
        return OrderResult(
            broker_order_id=broker_order_id,
            status=status,
            filled_qty=filled_qty,
            avg_fill_price=avg_price,
        )