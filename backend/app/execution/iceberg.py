"""
Iceberg execution: show only small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding true size.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger

# Maximum allowed price drift from the initial reference price (2%)
_MAX_PRICE_DRIFT: float = 0.02
# Minimum remaining quantity threshold to stop the loop
_MIN_REMAINING_QTY: float = 0.01
# Minimum fill threshold to consider the order as filled (95%)
_FILL_THRESHOLD: float = 0.95


class IcebergExecution:
    def __init__(
        self,
        broker: AbstractBroker,
        visible_pct: float = 0.10,
        refill_delay_seconds: int = 5,
    ) -> None:
        if not 0.0 < visible_pct <= 1.0:
            raise ValueError("visible_pct must be in (0, 1].")
        if refill_delay_seconds < 0:
            raise ValueError("refill_delay_seconds must be non‑negative.")
        self.broker: AbstractBroker = broker
        self.visible_pct: float = visible_pct
        self.refill_delay_seconds: int = refill_delay_seconds

    async def _fetch_initial_price(self, request: OrderRequest) -> Optional[float]:
        """Return a reference price for drift checks."""
        # Prefer an explicit price on the request (e.g., limit order)
        if getattr(request, "price", None):
            return float(request.price)  # type: ignore[arg-type]
        # Fall back to broker's market data if available
        if hasattr(self.broker, "get_last_price"):
            try:
                price = await self.broker.get_last_price(request.symbol)  # type: ignore[attr-defined]
                return float(price)
            except Exception as exc:  # pragma: no cover
                logger.debug("Unable to fetch market price", error=str(exc))
        return None

    async def execute(self, request: OrderRequest) -> OrderResult:
        """Execute an iceberg order with safety checks and drift monitoring."""
        if request.quantity <= 0:
            raise ValueError("Order quantity must be positive.")

        visible_qty: float = max(1.0, request.quantity * self.visible_pct)
        remaining: float = request.quantity
        total_filled: float = 0.0
        total_cost: float = 0.0
        last_result: Optional[OrderResult] = None

        # Reference price for drift detection
        reference_price: Optional[float] = await self._fetch_initial_price(request)

        while remaining > _MIN_REMAINING_QTY:
            # Optional market‑open guard
            if hasattr(self.broker, "is_market_open"):
                try:
                    if not await self.broker.is_market_open():  # type: ignore[attr-defined]
                        logger.info("Market closed, aborting iceberg execution")
                        break
                except Exception as exc:  # pragma: no cover
                    logger.debug("Market open check failed", error=str(exc))

            slice_qty: float = min(visible_qty, remaining)
            slice_req = OrderRequest(
                **{**asdict(request), "quantity": slice_qty, "order_type": "market"}
            )

            try:
                # Apply a timeout to each slice to avoid hanging indefinitely
                result: OrderResult = await asyncio.wait_for(
                    self.broker.place_order(slice_req), timeout=30
                )
                total_filled += result.filled_qty
                remaining -= result.filled_qty
                if result.avg_fill_price is not None:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result

                logger.debug(
                    "Iceberg slice executed",
                    filled=result.filled_qty,
                    remaining=remaining,
                    avg_price=result.avg_fill_price,
                )

                # Drift check – abort if price moves beyond allowed tolerance
                if reference_price is not None and total_filled > 0:
                    current_avg = total_cost / total_filled
                    drift = abs(current_avg - reference_price) / reference_price
                    if drift > _MAX_PRICE_DRIFT:
                        logger.warning(
                            "Price drift exceeded threshold",
                            drift=drift,
                            threshold=_MAX_PRICE_DRIFT,
                        )
                        break

                if remaining > _MIN_REMAINING_QTY:
                    await asyncio.sleep(self.refill_delay_seconds)
            except asyncio.TimeoutError:
                logger.warning("Iceberg slice timed out")
                break
            except Exception as exc:
                logger.warning("Iceberg slice failed", error=str(exc))
                break

        avg_price: Optional[float] = total_cost / total_filled if total_filled > 0 else None
        status: str
        if total_filled >= request.quantity * _FILL_THRESHOLD:
            status = "filled"
        elif total_filled > 0:
            status = "partial"
        else:
            status = "cancelled"

        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "iceberg",
            status=status,
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )