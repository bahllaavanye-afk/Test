"""
Iceberg execution: show only small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding true size.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult

# Import a specific broker exception if available; fall back to a generic one.
try:
    from app.brokers.base import BrokerError  # type: ignore
except ImportError:  # pragma: no cover
    class BrokerError(Exception):
        """Fallback broker error when specific exception class is unavailable."""
        pass

from app.utils.logging import logger


class IcebergExecution:
    """Execute an order using the iceberg strategy.

    The order is split into visible slices; each slice is sent as a market order.
    After each slice is filled, a delay is observed before the next slice is
    submitted. Errors are handled explicitly and logged with structured data.
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
        """Run the iceberg execution for the given order request.

        Args:
            request: The original order request containing quantity and other
                parameters.

        Returns:
            An :class:`OrderResult` summarising the total filled quantity,
            average fill price and execution status.
        """
        visible_qty: float = max(1.0, request.quantity * self.visible_pct)
        remaining: float = request.quantity
        total_filled: float = 0.0
        total_cost: float = 0.0
        last_result: Optional[OrderResult] = None
        error_occurred: bool = False

        while remaining > 0.01:
            slice_qty = min(visible_qty, remaining)
            slice_req = OrderRequest(
                **{**asdict(request), "quantity": slice_qty, "order_type": "market"}
            )
            try:
                result: OrderResult = await self.broker.place_order(slice_req)
                total_filled += result.filled_qty
                remaining -= result.filled_qty
                if result.avg_fill_price is not None:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result

                logger.debug(
                    "Iceberg slice executed",
                    filled=result.filled_qty,
                    remaining=remaining,
                    slice_qty=slice_qty,
                    broker_order_id=result.broker_order_id,
                )

                if remaining > 0.01:
                    await asyncio.sleep(self.refill_delay_seconds)

            except BrokerError as be:
                logger.error(
                    "Broker error during iceberg slice",
                    error=str(be),
                    slice_qty=slice_qty,
                    remaining=remaining,
                )
                error_occurred = True
                break
            except asyncio.TimeoutError as te:
                logger.error(
                    "Timeout while placing iceberg slice",
                    error=str(te),
                    slice_qty=slice_qty,
                    remaining=remaining,
                )
                error_occurred = True
                break
            except Exception as e:  # pragma: no cover
                logger.exception(
                    "Unexpected error during iceberg slice",
                    error=str(e),
                    slice_qty=slice_qty,
                    remaining=remaining,
                )
                error_occurred = True
                break

        avg_price: Optional[float] = total_cost / total_filled if total_filled > 0 else None
        status = "error" if error_occurred else (
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