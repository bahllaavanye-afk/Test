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
    """
    Handles execution of large orders by breaking them into smaller visible slices.
    """

    def __init__(self, broker: AbstractBroker, visible_pct: float = 0.10, refill_delay_seconds: int = 5):
        self.broker = broker
        self.visible_pct = visible_pct
        self.refill_delay_seconds = refill_delay_seconds

    async def execute(self, request: OrderRequest | None) -> OrderResult:
        """
        Execute an iceberg order.

        Args:
            request: The original order request. May be ``None`` or contain invalid
                quantity values. In such cases the method returns a rejected result
                without contacting the broker.

        Returns:
            OrderResult summarising the overall execution.
        """
        # ----- Edge‑case handling -------------------------------------------------
        if request is None:
            logger.warning("Iceberg execution received None request")
            return OrderResult(
                broker_order_id="none",
                status="rejected",
                filled_qty=0.0,
                avg_fill_price=None,
            )

        if request.quantity is None or request.quantity <= 0:
            logger.warning(
                "Iceberg execution received invalid quantity",
                quantity=request.quantity,
            )
            return OrderResult(
                broker_order_id="invalid_qty",
                status="rejected",
                filled_qty=0.0,
                avg_fill_price=None,
            )
        # -------------------------------------------------------------------------

        # Ensure visible quantity is at least 1 unit and respects the percentage.
        visible_qty = max(1.0, request.quantity * self.visible_pct)

        remaining = request.quantity
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None

        # Use a small epsilon to avoid infinite loops caused by floating‑point
        # rounding errors (off‑by‑one edge case).
        epsilon = 1e-6

        while remaining > epsilon:
            slice_qty = min(visible_qty, remaining)

            # Defensive copy – ``asdict`` works only on dataclasses; guard against
            # accidental mutation of the original request.
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

                # Only pause if there is still work to do; avoid unnecessary delay
                # when the remaining amount is below the epsilon.
                if remaining > epsilon:
                    await asyncio.sleep(self.refill_delay_seconds)
            except Exception as e:  # pragma: no cover
                logger.warning("Iceberg slice failed", error=str(e))
                break

        avg_price = total_cost / total_filled if total_filled > 0 else None
        status = (
            "filled"
            if total_filled >= request.quantity * 0.95
            else "partial"
        )
        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "iceberg",
            status=status,
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )