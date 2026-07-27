"""Result construction for the sliced execution algorithms.

TWAP, VWAP, iceberg and Almgren-Chriss all work the same way: split an order
into slices, fire them in sequence, tally what came back. They also all ended
the same way:

    return OrderResult(
        broker_order_id=last_result.broker_order_id if last_result else "vwap",
        status="filled" if total_filled >= request.quantity * 0.95 else "partial",
        filled_qty=total_filled,
        avg_fill_price=avg_price,
    )

When every slice failed that produced `filled_qty=0` with `status="partial"`
and a broker_order_id of `"vwap"` — a fabricated id no broker has ever issued,
attached to an execution that filled nothing, reported as a partial fill.
Callers cannot distinguish it from a genuine partial, and anything that tries
to poll or cancel that id fails.

This module centralises the ending so a total failure reads as one.
"""
from __future__ import annotations

from app.brokers.base import OrderRequest, OrderResult
from app.utils.logging import logger

# Fraction of the requested quantity that counts as a complete fill.
FILL_COMPLETE_RATIO = 0.95


def build_slice_result(
    algo: str,
    request: OrderRequest,
    *,
    total_filled: float,
    total_cost: float,
    last_result: OrderResult | None,
    slices_attempted: int = 0,
    slices_failed: int = 0,
    last_error: str | None = None,
) -> OrderResult:
    """Assemble the final OrderResult for a sliced execution.

    A run that filled nothing comes back as ``status="rejected"`` with an empty
    broker_order_id, not as a "partial" fill carrying a synthetic id.
    """
    avg_price = total_cost / total_filled if total_filled > 0 else None

    payload: dict = {
        "algo": algo,
        "requested_qty": request.quantity,
        "slices_attempted": slices_attempted,
        "slices_failed": slices_failed,
    }
    if last_error:
        payload["last_error"] = last_error

    if total_filled <= 0:
        logger.error(
            "%s execution filled NOTHING — no position was opened",
            algo,
            symbol=request.symbol,
            side=request.side,
            requested_qty=request.quantity,
            slices_attempted=slices_attempted,
            slices_failed=slices_failed,
            last_error=last_error,
        )
        return OrderResult(
            # No fill means no broker order to point at. Never invent an id.
            broker_order_id=last_result.broker_order_id if last_result else "",
            status="rejected",
            filled_qty=0.0,
            avg_fill_price=None,
            raw_payload=payload,
        )

    complete = total_filled >= request.quantity * FILL_COMPLETE_RATIO
    if not complete:
        logger.warning(
            "%s execution only partially filled",
            algo,
            symbol=request.symbol,
            requested_qty=request.quantity,
            filled_qty=total_filled,
            slices_failed=slices_failed,
        )

    return OrderResult(
        broker_order_id=last_result.broker_order_id if last_result else "",
        status="filled" if complete else "partial",
        filled_qty=total_filled,
        avg_fill_price=avg_price,
        raw_payload=payload,
    )
