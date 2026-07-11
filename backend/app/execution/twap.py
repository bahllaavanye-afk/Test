"""
TWAP (Time-Weighted Average Price) execution.
Splits large orders into N equal slices over duration minutes.
Minimizes market impact for large positions.
"""
import asyncio
import time
from dataclasses import asdict
from typing import Any

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.exceptions import BrokerError  # lives in utils, NOT brokers.base (#298 broke this import)
from app.utils.logging import logger


class TWAPExecution:
    def __init__(self, broker: AbstractBroker, slices: int = 10, duration_minutes: int = 30):
        self.broker = broker
        self.slices = slices
        self.sleep_seconds = (duration_minutes * 60) / slices

    async def execute(self, request: OrderRequest) -> OrderResult:
        """
        Execute a TWAP order.

        Parameters
        ----------
        request : OrderRequest
            The original order request to be split.

        Returns
        -------
        OrderResult
            Aggregated result of the TWAP execution.
        """
        start_ts = time.time()
        slice_qty = request.quantity / self.slices
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None
        consecutive_failures = 0
        successful_slices = 0

        for i in range(self.slices):
            slice_req = OrderRequest(
                **{**asdict(request), "quantity": slice_qty, "order_type": "market"}
            )
            try:
                result = await self.broker.place_order(slice_req)
                total_filled += result.filled_qty
                if result.avg_fill_price:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result
                consecutive_failures = 0
                successful_slices += 1
            except (BrokerError, ConnectionError, TimeoutError) as e:
                consecutive_failures += 1
                logger.warning(
                    f"TWAP slice {i + 1}/{self.slices} failed for {request.symbol}: {e}",
                    extra={"symbol": request.symbol, "slice": i + 1, "error": str(e)},
                )
                if consecutive_failures >= 3:
                    logger.error(
                        f"TWAP {request.symbol}: {consecutive_failures} consecutive failures — aborting",
                        extra={
                            "symbol": request.symbol,
                            "consecutive_failures": consecutive_failures,
                        },
                    )
                    break
            except Exception as e:
                # Unexpected exception: log full traceback and abort execution
                logger.exception(
                    f"Unexpected error during TWAP execution for {request.symbol} slice {i + 1}",
                    extra={"symbol": request.symbol, "slice": i + 1, "error": str(e)},
                )
                raise

            if i < self.slices - 1:
                await asyncio.sleep(self.sleep_seconds)

        avg_price = total_cost / total_filled if total_filled > 0 else None
        execution_time = time.time() - start_ts

        # Structured logging of key metrics
        logger.info(
            f"TWAP execution completed for {request.symbol}",
            extra={
                "symbol": request.symbol,
                "requested_quantity": request.quantity,
                "executed_quantity": total_filled,
                "average_fill_price": avg_price,
                "total_cost": total_cost,
                "slices_requested": self.slices,
                "slices_successful": successful_slices,
                "execution_time_seconds": execution_time,
                "status": "filled"
                if total_filled >= request.quantity * 0.95
                else "partial",
            },
        )

        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "twap",
            status="filled"
            if total_filled >= request.quantity * 0.95
            else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )