"""
Iceberg execution: show only a small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding the true size.
"""

import asyncio
from dataclasses import asdict
from typing import Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger


class IcebergExecution:
    """
    Executes an order using an iceberg strategy.

    The order is split into visible slices defined by ``visible_pct``.
    After each slice is filled (or partially filled), the next slice is sent
    after ``refill_delay_seconds``.  The strategy stops when the total
    remaining quantity is below a small threshold or an exception occurs.
    """

    def __init__(
        self,
        broker: AbstractBroker,
        visible_pct: float = 0.10,
        refill_delay_seconds: int = 5,
    ) -> None:
        if not 0 < visible_pct <= 1:
            raise ValueError("visible_pct must be between 0 (exclusive) and 1 (inclusive)")
        if refill_delay_seconds < 0:
            raise ValueError("refill_delay_seconds must be non‑negative")

        self.broker = broker
        self.visible_pct = visible_pct
        self.refill_delay_seconds = refill_delay_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        """
        Execute the iceberg order.

        Parameters
        ----------
        request: OrderRequest
            The original order request.

        Returns
        -------
        OrderResult
            Aggregated result of all slices.
        """
        # Minimum visible quantity is one unit; enforce a sensible floor.
        visible_qty: float = max(1.0, request.quantity * self.visible_pct)

        remaining: float = request.quantity
        total_filled: float = 0.0
        total_cost: float = 0.0
        last_result: Optional[OrderResult] = None

        # Continue until the remaining amount is negligible.
        while remaining > 0.01:
            slice_qty = min(visible_qty, remaining)
            slice_req = OrderRequest(
                **{**asdict(request), "quantity": slice_qty, "order_type": "market"}
            )
            try:
                result = await self.broker.place_order(slice_req)

                total_filled += result.filled_qty
                remaining -= result.filled_qty

                if result.avg_fill_price is not None:
                    total_cost += result.avg_fill_price * result.filled_qty

                last_result = result
                logger.debug(
                    "Iceberg slice executed",
                    filled=result.filled_qty,
                    remaining=remaining,
                )

                if remaining > 0.01:
                    await asyncio.sleep(self.refill_delay_seconds)

            except Exception as exc:
                logger.warning("Iceberg slice failed", error=str(exc))
                break

        avg_price: Optional[float] = (
            total_cost / total_filled if total_filled > 0 else None
        )

        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "iceberg",
            status="filled"
            if total_filled >= request.quantity * 0.95
            else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )