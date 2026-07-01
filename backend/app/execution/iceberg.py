"""
Iceberg execution: show only small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding true size.
"""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import asdict
from typing import Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult

# Optional import of a broker‑specific error class. If it does not exist we fall back to a generic Exception.
try:
    from app.brokers.base import BrokerError  # type: ignore
except Exception:  # noqa: BLE001
    BrokerError = Exception  # type: ignore

from app.utils.logging import logger


class IcebergExecution:
    """
    Execute an order using the iceberg strategy.

    The order is split into visible slices defined by ``visible_pct``. After each slice
    is filled (or partially filled) the remaining quantity is sent as a new slice
    after ``refill_delay_seconds`` seconds. Errors from the broker are caught, logged,
    and cause the execution loop to terminate gracefully.
    """

    def __init__(self, broker: AbstractBroker, visible_pct: float = 0.10, refill_delay_seconds: int = 5):
        """
        Initialise the iceberg executor.

        Args:
            broker: An instance of :class:`AbstractBroker` used to place orders.
            visible_pct: Fraction of the total quantity to expose per slice (default 10%).
            refill_delay_seconds: Seconds to wait between slices (default 5).
        """
        self.broker = broker
        self.visible_pct = visible_pct
        self.refill_delay_seconds = refill_delay_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        """
        Execute the given ``OrderRequest`` using the iceberg strategy.

        Args:
            request: The original order request to be split into slices.

        Returns:
            An :class:`OrderResult` summarising the overall execution.
        """
        visible_qty = max(1.0, request.quantity * self.visible_pct)
        remaining = request.quantity
        total_filled: float = 0.0
        total_cost: float = 0.0
        last_result: Optional[OrderResult] = None
        slice_index: int = 0

        while remaining > 0.01:
            slice_index += 1
            slice_qty = min(visible_qty, remaining)
            slice_req = OrderRequest(
                **{**asdict(request), "quantity": slice_qty, "order_type": "market"}
            )
            try:
                result = await self.broker.place_order(slice_req)
                total_filled += result.filled_qty
                remaining -= result.filled_qty
                if result.avg_fill_price:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result

                logger.debug(
                    "Iceberg slice executed",
                    slice_index=slice_index,
                    slice_qty=slice_qty,
                    filled_qty=result.filled_qty,
                    remaining_qty=remaining,
                )

                if remaining > 0.01:
                    await asyncio.sleep(self.refill_delay_seconds)

            except BrokerError as be:
                logger.error(
                    "BrokerError during iceberg slice",
                    slice_index=slice_index,
                    slice_qty=slice_qty,
                    error_type=type(be).__name__,
                    error_message=str(be),
                )
                break
            except asyncio.TimeoutError as te:
                logger.error(
                    "TimeoutError during iceberg slice",
                    slice_index=slice_index,
                    slice_qty=slice_qty,
                    timeout_seconds=self.refill_delay_seconds,
                    error_message=str(te),
                )
                break
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Unexpected error during iceberg execution",
                    slice_index=slice_index,
                    slice_qty=slice_qty,
                    remaining_qty=remaining,
                    traceback=traceback.format_exc(),
                )
                break

        avg_price = total_cost / total_filled if total_filled > 0 else None
        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "iceberg",
            status="filled"
            if total_filled >= request.quantity * 0.95
            else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )