"""
Iceberg execution: show only small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding true size.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, BrokerError
from app.utils.logging import logger


class IcebergExecution:
    def __init__(
        self,
        broker: AbstractBroker,
        visible_pct: float = 0.10,
        refill_delay_seconds: int = 5,
    ) -> None:
        if not 0 < visible_pct <= 1:
            raise ValueError("visible_pct must be > 0 and <= 1")
        if refill_delay_seconds < 0:
            raise ValueError("refill_delay_seconds must be non‑negative")
        self.broker = broker
        self.visible_pct = visible_pct
        self.refill_delay_seconds = refill_delay_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        """Execute an iceberg order by submitting successive market slices.

        Args:
            request: The original order request containing the total quantity.

        Returns:
            An OrderResult summarising the cumulative execution.
        """
        visible_qty = max(1.0, request.quantity * self.visible_pct)
        remaining = request.quantity
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None

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
                    slice_quantity=slice_qty,
                    filled=result.filled_qty,
                    remaining=remaining,
                )
                if remaining > 0.01:
                    await asyncio.sleep(self.refill_delay_seconds)
            except BrokerError as be:
                logger.error(
                    "BrokerError during iceberg slice",
                    error=str(be),
                    slice_quantity=slice_qty,
                    request_id=getattr(request, "client_order_id", None),
                )
                break
            except asyncio.TimeoutError as te:
                logger.error(
                    "Timeout while placing iceberg slice",
                    error=str(te),
                    slice_quantity=slice_qty,
                )
                break
            except Exception as e:  # pylint: disable=broad-except
                logger.exception(
                    "Unexpected error during iceberg execution",
                    error=str(e),
                    slice_quantity=slice_qty,
                )
                break

        avg_price = total_cost / total_filled if total_filled > 0 else None
        return OrderResult(
            broker_order_id=(
                last_result.broker_order_id if last_result else "iceberg"
            ),
            status=(
                "filled"
                if total_filled >= request.quantity * 0.95
                else "partial"
            ),
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )