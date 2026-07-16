"""
Iceberg execution: show only a small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding the true size.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult

logger = logging.getLogger(__name__)


class IcebergExecution:
    """
    Executes an order using an iceberg strategy.

    Parameters
    ----------
    broker: AbstractBroker
        Broker used to place slice orders.
    visible_pct: float, optional
        Percentage of the total order quantity to expose per slice.
        Must be in the interval (0, 1]. Default is 0.10 (10 %).
    refill_delay_seconds: int, optional
        Seconds to wait between consecutive slices. Default is 5.
    max_slices: int | None, optional
        Optional hard limit on the number of slices. ``None`` means unlimited.
    """

    def __init__(
        self,
        broker: AbstractBroker,
        visible_pct: float = 0.10,
        refill_delay_seconds: int = 5,
        max_slices: Optional[int] = None,
    ) -> None:
        if not (0 < visible_pct <= 1):
            raise ValueError("visible_pct must be > 0 and ≤ 1")
        if refill_delay_seconds < 0:
            raise ValueError("refill_delay_seconds must be non‑negative")
        self.broker = broker
        self.visible_pct = visible_pct
        self.refill_delay_seconds = refill_delay_seconds
        self.max_slices = max_slices
        self._lock = asyncio.Lock()

    async def execute(self, request: OrderRequest) -> OrderResult:
        """
        Execute the given order request using an iceberg approach.

        The method breaks the total quantity into slices, each slice being a
        market order of size ``visible_qty`` (or the remaining quantity for the
        final slice). After each successful slice, the method sleeps for
        ``refill_delay_seconds`` before placing the next slice.

        The execution stops when:

        * The total filled quantity reaches at least 95 % of the original
          request quantity.
        * The remaining quantity falls below a negligible threshold (0.01).
        * The broker raises an exception for a slice.
        * An optional ``max_slices`` limit is reached.

        Returns
        -------
        OrderResult
            Aggregated result of all slices.
        """
        async with self._lock:
            visible_qty = max(1.0, request.quantity * self.visible_pct)
            remaining = request.quantity
            total_filled = 0.0
            total_cost = 0.0
            last_result: Optional[OrderResult] = None
            slice_counter = 0

            # Confirmation filter: ensure the market is not halted and the broker
            # provides a valid price before starting the iceberg.
            if not await self._can_start_iceberg(request):
                logger.warning("Iceberg execution aborted – market conditions not met")
                return OrderResult(
                    broker_order_id="iceberg_aborted",
                    status="rejected",
                    filled_qty=0.0,
                    avg_fill_price=None,
                )

            while remaining > 0.01:
                if self.max_slices is not None and slice_counter >= self.max_slices:
                    logger.info(
                        "Iceberg slice limit reached",
                        slice_counter=slice_counter,
                        max_slices=self.max_slices,
                    )
                    break

                slice_qty = min(visible_qty, remaining)
                slice_req = OrderRequest(
                    **{**asdict(request), "quantity": slice_qty, "order_type": "market"}
                )

                try:
                    result = await self.broker.place_order(slice_req)
                except Exception as exc:  # pragma: no cover
                    logger.warning(
                        "Iceberg slice failed",
                        error=str(exc),
                        slice_qty=slice_qty,
                        remaining=remaining,
                    )
                    break

                # Update aggregates
                filled = result.filled_qty or 0.0
                total_filled += filled
                remaining -= filled
                if result.avg_fill_price is not None:
                    total_cost += result.avg_fill_price * filled

                last_result = result
                slice_counter += 1

                logger.debug(
                    "Iceberg slice executed",
                    slice_qty=slice_qty,
                    filled=filled,
                    remaining=remaining,
                    slice_index=slice_counter,
                )

                # Early exit if we have filled enough of the original order
                if total_filled >= request.quantity * 0.95:
                    logger.info(
                        "Iceberg reached 95 % fill target",
                        total_filled=total_filled,
                        target=request.quantity * 0.95,
                    )
                    break

                if remaining > 0.01:
                    await asyncio.sleep(self.refill_delay_seconds)

            avg_price = total_cost / total_filled if total_filled > 0 else None
            status = "filled" if total_filled >= request.quantity * 0.95 else "partial"

            return OrderResult(
                broker_order_id=last_result.broker_order_id if last_result else "iceberg",
                status=status,
                filled_qty=total_filled,
                avg_fill_price=avg_price,
            )

    async def _can_start_iceberg(self, request: OrderRequest) -> bool:
        """
        Confirmation filter to decide whether the iceberg execution should start.

        Currently checks that the broker reports a non‑zero best bid/ask spread.
        This method can be extended with additional market‑state checks without
        impacting the main execution flow.

        Returns
        -------
        bool
            ``True`` if the market appears liquid enough to start an iceberg,
            ``False`` otherwise.
        """
        try:
            market_data = await self.broker.get_market_snapshot(request.symbol)
            spread = getattr(market_data, "ask_price", 0) - getattr(
                market_data, "bid_price", 0
            )
            if spread <= 0:
                logger.debug("Zero or negative spread detected", spread=spread)
                return False
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to fetch market snapshot for iceberg confirmation", error=str(exc))
            return False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(broker={self.broker!r}, "
            f"visible_pct={self.visible_pct}, refill_delay_seconds={self.refill_delay_seconds}, "
            f"max_slices={self.max_slices})"
        )