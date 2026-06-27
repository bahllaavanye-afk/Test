"""
Iceberg execution: show only small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding true size.
"""
from __future__ import annotations

import asyncio

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger


class IcebergExecution:
    """Execute orders using the iceberg strategy.

    The strategy splits a large order into smaller visible slices. After each slice
    is filled, the next slice is submitted after a configurable delay.
    """

    def __init__(self, broker: AbstractBroker, visible_pct: float = 0.10, refill_delay_seconds: int = 5):
        """
        Args:
            broker: Broker implementation used to place orders.
            visible_pct: Fraction of the total quantity to expose per slice (0 < pct <= 1).
            refill_delay_seconds: Seconds to wait between submitting slices.
        """
        self.broker = broker
        self.visible_pct = visible_pct
        self.refill_delay_seconds = refill_delay_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        """Execute the iceberg order.

        Args:
            request: The original order request.

        Returns:
            An OrderResult summarising the aggregated execution.
        """
        visible_qty = max(1.0, request.quantity * self.visible_pct)
        remaining = request.quantity
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None

        while remaining > 0.01:
            slice_qty = min(visible_qty, remaining)
            slice_req = OrderRequest(
                **{**request.__dict__, "quantity": slice_qty, "order_type": "market"}
            )
            try:
                result = await self.broker.place_order(slice_req)
                filled = result.filled_qty or 0.0
                total_filled += filled
                remaining -= filled
                if result.avg_fill_price is not None:
                    total_cost += result.avg_fill_price * filled
                last_result = result
                logger.debug("Iceberg slice", filled=filled, remaining=remaining)

                if filled == 0:
                    logger.warning("Iceberg slice filled zero quantity, aborting to avoid infinite loop")
                    break

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