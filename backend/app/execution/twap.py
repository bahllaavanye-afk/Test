"""
TWAP (Time-Weighted Average Price) execution module.

This module provides a simple TWAP execution strategy that splits a large
order into a configurable number of equal slices and executes each slice
at regular intervals. The goal is to minimise market impact by spreading
the order execution over a defined duration.
"""

import asyncio
from dataclasses import asdict
from typing import Any, Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.exceptions import BrokerError  # lives in utils, NOT brokers.base (#298 broke this import)
from app.utils.logging import logger


class TWAPExecution:
    """Execute orders using a Time‑Weighted Average Price (TWAP) strategy.

    The strategy divides a parent order into ``slices`` equal parts and
    dispatches each part at a fixed interval determined by ``duration_minutes``.
    It aggregates the results of each slice into a single :class:`OrderResult`.

    Attributes
    ----------
    broker : AbstractBroker
        Broker implementation used to place individual slice orders.
    slices : int
        Number of slices the parent order will be split into.
    sleep_seconds : float
        Pause duration (in seconds) between consecutive slice executions.
    """

    def __init__(self, broker: AbstractBroker, slices: int = 10, duration_minutes: int = 30) -> None:
        """
        Initialise the TWAP execution instance.

        Parameters
        ----------
        broker : AbstractBroker
            The broker used for order placement.
        slices : int, optional
            Number of equal slices to split the order into (default is 10).
        duration_minutes : int, optional
            Total duration over which the slices will be executed (default is 30 minutes).
        """
        self.broker = broker
        self.slices = slices
        self.sleep_seconds = (duration_minutes * 60) / slices

    async def execute(self, request: OrderRequest) -> OrderResult:
        """
        Execute a TWAP order.

        The order request is divided into ``self.slices`` equal parts. Each part
        is sent to the broker as a market order. The method tracks filled quantity,
        average fill price and aborts if three consecutive slice submissions fail.

        Parameters
        ----------
        request : OrderRequest
            The original order request to be split.

        Returns
        -------
        OrderResult
            Aggregated result of the TWAP execution, containing the total filled
            quantity, average fill price, and an overall status.
        """
        slice_qty = request.quantity / self.slices
        total_filled = 0.0
        total_cost = 0.0
        last_result: Optional[OrderResult] = None
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