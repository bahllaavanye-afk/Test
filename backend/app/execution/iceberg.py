"""
Iceberg execution: show only small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding true size.
"""
from __future__ import annotations
import asyncio
from dataclasses import asdict
from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.execution.slice_result import build_slice_result
from app.utils.logging import logger


class IcebergExecution:
    def __init__(self, broker: AbstractBroker, visible_pct: float = 0.10, refill_delay_seconds: int = 5):
        self.broker = broker
        self.visible_pct = visible_pct
        self.refill_delay_seconds = refill_delay_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        visible_qty = max(1.0, request.quantity * self.visible_pct)
        remaining = request.quantity
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None
        slices_attempted = 0
        slices_failed = 0
        last_error: str | None = None

        while remaining > 0.01:
            slice_qty = min(visible_qty, remaining)
            slice_req = OrderRequest(
                **{**asdict(request), "quantity": slice_qty, "order_type": "market"}
            )
            slices_attempted += 1
            try:
                result = await self.broker.place_order(slice_req)
                total_filled += result.filled_qty
                remaining -= result.filled_qty
                if result.avg_fill_price:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result
                logger.debug("Iceberg slice", filled=result.filled_qty, remaining=remaining)

                # An accepted-but-unfilled slice returns filled_qty=0, which leaves
                # `remaining` untouched — this loop would spin on it forever.
                if result.filled_qty <= 0:
                    logger.warning(
                        "Iceberg slice accepted but filled nothing — stopping to "
                        "avoid an unbounded refill loop",
                        symbol=request.symbol,
                        remaining=remaining,
                    )
                    break

                if remaining > 0.01:
                    await asyncio.sleep(self.refill_delay_seconds)
            except Exception as e:
                slices_failed += 1
                last_error = str(e)
                logger.warning("Iceberg slice failed", error=last_error)
                break

        return build_slice_result(
            "Iceberg", request,
            total_filled=total_filled,
            total_cost=total_cost,
            last_result=last_result,
            slices_attempted=slices_attempted,
            slices_failed=slices_failed,
            last_error=last_error,
        )
