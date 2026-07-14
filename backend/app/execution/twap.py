"""
TWAP (Time-Weighted Average Price) execution.
Splits large orders into N equal slices over duration minutes.
Minimizes market impact for large positions.
"""
import asyncio
from dataclasses import asdict
from typing import Any, Optional

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
        slice_qty = request.quantity / self.slices
        total_filled = 0.0
        total_cost = 0.0
        last_result: Optional[OrderResult] = None
        consecutive_failures = 0

        for i in range(self.slices):
            slice_req = self._create_slice_request(request, slice_qty)

            try:
                result = await self.broker.place_order(slice_req)
                total_filled, total_cost, last_result = self._process_success(
                    result, total_filled, total_cost
                )
                consecutive_failures = 0
            except (BrokerError, ConnectionError, TimeoutError) as e:
                consecutive_failures = self._handle_known_failure(
                    e, request, i, consecutive_failures
                )
                if consecutive_failures >= 3:
                    break
            except Exception as e:
                # Unexpected exception: log full traceback and abort execution
                logger.exception(
                    f"Unexpected error during TWAP execution for {request.symbol} slice {i + 1}",
                    extra={"symbol": request.symbol, "slice": i + 1, "error": str(e)},
                )
                raise

            if i < self.slices - 1:
                await self._maybe_sleep()

        avg_price = total_cost / total_filled if total_filled > 0 else None
        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "twap",
            status="filled"
            if total_filled >= request.quantity * 0.95
            else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )

    def _create_slice_request(self, original: OrderRequest, slice_qty: float) -> OrderRequest:
        """
        Build a slice OrderRequest based on the original request.
        """
        return OrderRequest(
            **{**asdict(original), "quantity": slice_qty, "order_type": "market"}
        )

    def _process_success(
        self,
        result: OrderResult,
        total_filled: float,
        total_cost: float,
    ) -> tuple[float, float, OrderResult]:
        """
        Update aggregation metrics after a successful slice execution.
        """
        total_filled += result.filled_qty
        if result.avg_fill_price:
            total_cost += result.avg_fill_price * result.filled_qty
        return total_filled, total_cost, result

    def _handle_known_failure(
        self,
        exc: Exception,
        request: OrderRequest,
        slice_index: int,
        current_failures: int,
    ) -> int:
        """
        Log a known failure (BrokerError, ConnectionError, TimeoutError) and
        return the updated consecutive failure count.
        """
        current_failures += 1
        logger.warning(
            f"TWAP slice {slice_index + 1}/{self.slices} failed for {request.symbol}: {exc}",
            extra={"symbol": request.symbol, "slice": slice_index + 1, "error": str(exc)},
        )
        if current_failures >= 3:
            logger.error(
                f"TWAP {request.symbol}: {current_failures} consecutive failures — aborting",
                extra={"symbol": request.symbol, "consecutive_failures": current_failures},
            )
        return current_failures

    async def _maybe_sleep(self) -> None:
        """
        Sleep for the configured interval between slices.
        """
        await asyncio.sleep(self.sleep_seconds)