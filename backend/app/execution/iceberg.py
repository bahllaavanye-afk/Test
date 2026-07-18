"""
Iceberg execution: show only small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding true size.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger


class IcebergExecution:
    def __init__(self, broker: AbstractBroker, visible_pct: float = 0.10, refill_delay_seconds: int = 5):
        self.broker = broker
        self.visible_pct = visible_pct
        self.refill_delay_seconds = refill_delay_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        start_time = time.perf_counter()

        visible_qty = max(1.0, request.quantity * self.visible_pct)
        remaining = request.quantity
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None
        slice_count = 0

        while remaining > 0.01:
            slice_qty = min(visible_qty, remaining)
            slice_req = OrderRequest(
                **{**asdict(request), "quantity": slice_qty, "order_type": "market"}
            )
            try:
                result = await self.broker.place_order(slice_req)
                slice_count += 1
                total_filled += result.filled_qty
                remaining -= result.filled_qty
                if result.avg_fill_price:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result
                logger.debug(
                    "Iceberg slice executed",
                    filled=result.filled_qty,
                    remaining=remaining,
                    slice_qty=slice_qty,
                    slice_number=slice_count,
                )

                if remaining > 0.01:
                    await asyncio.sleep(self.refill_delay_seconds)
            except Exception as e:
                logger.warning("Iceberg slice failed", error=str(e))
                break

        avg_price = total_cost / total_filled if total_filled > 0 else None
        execution_time = time.perf_counter() - start_time
        pnl = (avg_price * total_filled - total_cost) if avg_price is not None else None  # placeholder P&L calculation

        logger.info(
            "Iceberg execution completed",
            signal_count=slice_count,
            execution_time_seconds=execution_time,
            total_filled_qty=total_filled,
            avg_fill_price=avg_price,
            total_cost=total_cost,
            pnl=pnl,
        )

        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "iceberg",
            status="filled" if total_filled >= request.quantity * 0.95 else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )