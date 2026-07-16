"""
TWAP (Time-Weighted Average Price) execution.
Splits large orders into N equal slices over duration minutes.
Minimizes market impact for large positions.
"""
import asyncio
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, Field, validator

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.exceptions import BrokerError  # lives in utils, NOT brokers.base (#298 broke this import)
from app.utils.logging import logger


class TWAPParameters(BaseModel):
    """
    Configuration parameters for a TWAP execution.

    Attributes
    ----------
    slices : int
        Number of equal slices to split the order into.
        Must be a positive integer (≥ 1). Example: 10.
    duration_minutes : int
        Total duration in minutes over which the slices are executed.
        Must be a positive integer. Example: 30.
    """
    slices: int = Field(
        ...,
        description="Number of equal slices to split the order into.",
        ge=1,
        example=10,
    )
    duration_minutes: int = Field(
        ...,
        description="Total duration in minutes for the TWAP execution.",
        gt=0,
        example=30,
    )

    @validator("slices")
    def _validate_slices(cls, v: int) -> int:
        if v < 1:
            raise ValueError("slices must be at least 1")
        return v

    @validator("duration_minutes")
    def _validate_duration(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("duration_minutes must be positive")
        return v


class TWAPExecution:
    def __init__(self, broker: AbstractBroker, slices: int = 10, duration_minutes: int = 30):
        """
        Initialise a TWAP execution engine.

        Parameters
        ----------
        broker : AbstractBroker
            Broker implementation used to place individual slice orders.
        slices : int, optional
            Number of equal slices to split the order into. Default is 10.
        duration_minutes : int, optional
            Total duration in minutes for the TWAP execution. Default is 30.
        """
        params = TWAPParameters(slices=slices, duration_minutes=duration_minutes)
        self.broker = broker
        self.slices = params.slices
        self.sleep_seconds = (params.duration_minutes * 60) / params.slices

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
        last_result: OrderResult | None = None
        consecutive_failures = 0

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
            except (BrokerError, ConnectionError, TimeoutError) as e:
                consecutive_failures += 1
                logger.warning(
                    f"TWAP slice {i + 1}/{self.slices} failed for {request.symbol}: {e}",
                    extra={"symbol": request.symbol, "slice": i + 1, "error": str(e)},
                )
                if consecutive_failures >= 3:
                    logger.error(
                        f"TWAP {request.symbol}: {consecutive_failures} consecutive failures — aborting",
                        extra={"symbol": request.symbol, "consecutive_failures": consecutive_failures},
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
        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "twap",
            status="filled"
            if total_filled >= request.quantity * 0.95
            else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )