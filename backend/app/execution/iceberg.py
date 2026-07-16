"""
Iceberg execution: show only a small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding the true size.

This module provides the :class:`IcebergExecution` class which splits a large order
into a series of smaller market orders (the "iceberg slices").  The implementation
includes stricter validation of input parameters, additional safety checks
during execution, and a more robust determination of the final order status.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger


class IcebergExecution:
    """
    Execute a large order using an iceberg strategy.

    Parameters
    ----------
    broker: AbstractBroker
        Broker instance used to place individual slices.
    visible_pct: float, optional
        Fraction of the total quantity to expose in each slice. Must be in
        ``(0, 1]``. Default is ``0.10`` (10 %).
    refill_delay_seconds: int, optional
        Seconds to wait between consecutive slices. Must be non‑negative.
        Default is ``5`` seconds.
    """

    def __init__(
        self,
        broker: AbstractBroker,
        visible_pct: float = 0.10,
        refill_delay_seconds: int = 5,
    ) -> None:
        if not (0 < visible_pct <= 1):
            raise ValueError("visible_pct must be > 0 and <= 1")
        if refill_delay_seconds < 0:
            raise ValueError("refill_delay_seconds must be non‑negative")

        self.broker: AbstractBroker = broker
        self.visible_pct: float = visible_pct
        self.refill_delay_seconds: int = refill_delay_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        """
        Perform the iceberg execution.

        The method validates the request, splits it into slices according to
        ``visible_pct``, and places each slice as a market order.  Execution stops
        when the remaining quantity falls below a small tolerance, when an
        exception occurs, or when a safety slice‑count limit is reached.

        Parameters
        ----------
        request: OrderRequest
            The original order to be executed.

        Returns
        -------
        OrderResult
            Aggregated result of all slices.
        """
        if request.quantity <= 0:
            raise ValueError("Order quantity must be positive")

        # Determine the visible quantity per slice; enforce a minimum of 1 unit.
        visible_qty = max(1.0, request.quantity * self.visible_pct)

        remaining: float = request.quantity
        total_filled: float = 0.0
        total_cost: float = 0.0
        last_result: Optional[OrderResult] = None

        # Safety limit: avoid infinite loops due to unexpected broker behaviour.
        max_slices = int(request.quantity // visible_qty) + 10
        slice_counter = 0
        tolerance = 0.01  # quantity tolerance for completion

        while remaining > tolerance:
            if slice_counter >= max_slices:
                logger.warning(
                    "Iceberg execution aborted",
                    reason="max slice count exceeded",
                    slices=slice_counter,
                )
                break

            slice_qty = min(visible_qty, remaining)
            slice_req = OrderRequest(
                **{**asdict(request), "quantity": slice_qty, "order_type": "market"}
            )

            try:
                result = await self.broker.place_order(slice_req)
                filled = result.filled_qty or 0.0
                total_filled += filled
                remaining -= filled

                if result.avg_fill_price:
                    total_cost += result.avg_fill_price * filled

                last_result = result
                logger.debug(
                    "Iceberg slice executed",
                    filled=filled,
                    remaining=remaining,
                    slice_number=slice_counter + 1,
                )

                if remaining > tolerance and self.refill_delay_seconds:
                    await asyncio.sleep(self.refill_delay_seconds)

            except Exception as e:  # pragma: no cover
                logger.warning("Iceberg slice failed", error=str(e))
                break

            slice_counter += 1

        avg_price = total_cost / total_filled if total_filled > 0 else None
        status = (
            "filled"
            if total_filled >= request.quantity * 0.95
            else "partial"
        )

        return OrderResult(
            broker_order_id=(
                last_result.broker_order_id if last_result else "iceberg"
            ),
            status=status,
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )