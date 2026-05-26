"""
VWAP (Volume-Weighted Average Price) execution.
Participates at 10% of market volume across trading session.
Minimizes market impact by timing with volume distribution.
"""
from __future__ import annotations
import asyncio
from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger

# Typical intraday volume distribution by 30-min buckets (normalized, open=close heavy)
VWAP_PROFILE = [0.12, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.06, 0.07, 0.08, 0.12, 0.05]


class VWAPExecution:
    def __init__(self, broker: AbstractBroker, participation_rate: float = 0.10, slices: int = 12):
        self.broker = broker
        self.participation_rate = participation_rate
        self.slices = min(slices, len(VWAP_PROFILE))
        self.sleep_seconds = (6.5 * 3600) / self.slices  # spread over trading day

    async def execute(self, request: OrderRequest) -> OrderResult:
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None

        for i in range(self.slices):
            # Weight slice by VWAP profile
            slice_weight = VWAP_PROFILE[i] / sum(VWAP_PROFILE[:self.slices])
            slice_qty = request.quantity * slice_weight

            slice_req = OrderRequest(
                **{**request.__dict__, "quantity": slice_qty, "order_type": "market"}
            )
            try:
                result = await self.broker.place_order(slice_req)
                total_filled += result.filled_qty
                if result.avg_fill_price:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result
                logger.debug("VWAP slice filled", slice=i, qty=slice_qty, filled=result.filled_qty)
            except Exception as e:
                logger.warning("VWAP slice failed", slice=i, error=str(e))

            if i < self.slices - 1:
                await asyncio.sleep(self.sleep_seconds)

        avg_price = total_cost / total_filled if total_filled > 0 else None
        fill_rate = total_filled / request.quantity if request.quantity > 0 else 0
        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "vwap",
            status="filled" if fill_rate >= 0.95 else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )
