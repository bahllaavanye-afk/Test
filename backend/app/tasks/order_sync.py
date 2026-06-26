"""
Order sync task: polls broker every 15 seconds, detects new fills,
broadcasts via WebSocket, and posts to AgentBus risk channel.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field, validator

from app.utils.logging import logger

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore


class FillEventSchema(BaseModel):
    """Schema for fill events broadcast to WebSocket listeners."""

    type: str = Field(
        "fill",
        description="Event type identifier.",
        examples=["fill"],
    )
    order_id: str = Field(
        ...,
        description="Unique identifier of the order.",
        examples=["12345"],
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol of the filled order.",
        examples=["AAPL"],
    )
    filled_qty: float = Field(
        ...,
        ge=0,
        description="Quantity filled for the order.",
        examples=[10.0],
    )
    filled_avg_price: float = Field(
        ...,
        ge=0,
        description="Average price at which the order was filled.",
        examples=[150.25],
    )

    @validator("filled_qty", "filled_avg_price")
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non‑negative")
        return v


class PositionUpdateSchema(BaseModel):
    """Schema for position update events broadcast to WebSocket listeners."""

    type: str = Field(
        "update",
        description="Event type identifier.",
        examples=["update"],
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol whose position was updated.",
        examples=["AAPL"],
    )


async def sync_orders_once(db_session_factory) -> None:
    """
    Single tick of order sync logic.
    - Fetches open orders from DB + broker
    - Detects new fills (status changed to 'filled' or filled_qty increased)
    - Updates DB
    - Broadcasts fill events via WebSocket
    - Posts to AgentBus risk channel
    """
    from sqlalchemy import select

    from app.brokers.alpaca_orders import _base_url, _headers
    from app.models.account import Account
    from app.models.order import Order
    from app.ws.manager import manager

    if httpx is None:
        logger.warning("httpx not available — order sync skipped")
        return

    # ------------------------------------------------------------------
    # 1️⃣ Load pending orders together with their accounts
    # ------------------------------------------------------------------
    try:
        async with db_session_factory() as db:
            result = await db.execute(
                select(Order, Account)
                .join(Account, Order.account_id == Account.id)
                .where(
                    Order.status.in_(["pending", "accepted", "partially_filled", "new"]),
                    Account.is_active.is_(True),  # noqa: E712
                )
            )
            rows = result.all()
    except Exception as exc:  # pragma: no cover
        logger.debug("Order sync: DB fetch failed", error=str(exc))
        return

    if not rows:
        return

    # ------------------------------------------------------------------
    # 2️⃣ Group orders by account to reuse HTTP client & headers
    # ------------------------------------------------------------------
    account_orders: Dict[int, List[Tuple[Order, Account]]] = defaultdict(list)
    for order_obj, acct in rows:
        if not order_obj.broker_order_id or acct.broker != "alpaca":
            continue
        account_orders[acct.id].append((order_obj, acct))

    if not account_orders:
        return

    new_fills: List[Tuple[Order, int | None]] = []
    # ------------------------------------------------------------------
    # 3️⃣ Process each account batch
    # ------------------------------------------------------------------
    for acct_id, order_acct_list in account_orders.items():
        order_obj, acct = order_acct_list[0]  # any element gives us the account
        try:
            headers = await _headers(acct)
            base_url = _base_url(acct)
        except Exception as exc:  # pragma: no cover
            logger.debug(
                "Order sync: failed to prepare request data",
                account_id=acct_id,
                error=str(exc),
            )
            continue

        async with httpx.AsyncClient(timeout=8) as client:
            for order_obj, acct in order_acct_list:
                try:
                    resp = await client.get(
                        f"{base_url}/v2/orders/{order_obj.broker_order_id}",
                        headers=headers,
                    )
                except Exception as exc:  # pragma: no cover
                    logger.debug(
                        "Order sync: HTTP request failed",
                        order_id=order_obj.id,
                        error=str(exc),
                    )
                    continue

                if resp.status_code != 200:
                    logger.debug(
                        "Order sync: non‑200 response",
                        order_id=order_obj.id,
                        status_code=resp.status_code,
                    )
                    continue

                data = resp.json()
                new_status = data.get("status", order_obj.status)
                new_filled_qty = float(data.get("filled_qty") or 0)
                new_avg_price = (
                    float(data["filled_avg_price"])
                    if data.get("filled_avg_price")
                    else None
                )

                # Preserve previous values for fill detection
                prev_filled_qty = float(order_obj.filled_qty or 0)
                prev_status = order_obj.status

                # Apply updates directly on the ORM instance
                order_obj.status = new_status
                order_obj.filled_qty = new_filled_qty
                order_obj.avg_fill_price = new_avg_price

                # Detect new fill
                is_new_fill = (
                    (new_status == "filled" and prev_status != "filled")
                    or (new_filled_qty > prev_filled_qty and new_filled_qty > 0)
                )
                if is_new_fill:
                    new_fills.append((order_obj, getattr(acct, "user_id", None)))

    # ------------------------------------------------------------------
    # 4️⃣ Commit all changes in a single transaction
    # ------------------------------------------------------------------
    if new_fills:
        try:
            async with db_session_factory() as db:
                await db.commit()
            logger.info("Order sync complete", updated=len(new_fills))
        except Exception as exc:  # pragma: no cover
            logger.error("Order sync DB commit failed", error=str(exc))
            return

    # ------------------------------------------------------------------
    # 5️⃣ Broadcast fill events
    # ------------------------------------------------------------------
    for order, user_id in new_fills:
        try:
            uid = str(user_id) if user_id else "system"
            fill_payload = FillEventSchema(
                order_id=str(order.id),
                symbol=order.symbol,
                filled_qty=float(order.filled_qty or 0),
                filled_avg_price=float(order.avg_fill_price or 0),
            )
            await manager.broadcast(f"orders:{uid}", fill_payload.dict())
            position_payload = PositionUpdateSchema(symbol=order.symbol)
            await manager.broadcast(f"positions:{uid}", position_payload.dict())
        except Exception as exc:  # pragma: no cover
            logger.debug(
                "Order sync: WS broadcast failed",
                order_id=order.id,
                error=str(exc),
            )

        try:
            from app.tasks.agent_bus import get_bus

            bus = get_bus()
            await bus.post_finding(
                "risk",
                f"Fill: {order.symbol} {order.filled_qty} @ {order.avg_fill_price}",
                {"order_id": str(order.id)},
                from_agent="order_sync",
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Order sync: AgentBus post failed", order_id=order.id, error=str(exc))


async def run_order_sync_loop(db_session_factory, interval_seconds: int = 15) -> None:
    """
    Continuous loop: sync orders every `interval_seconds` seconds.
    Designed to run as a supervised asyncio task.
    """
    logger.info("Order sync loop started", interval_seconds=interval_seconds)
    while True:
        try:
            await sync_orders_once(db_session_factory)
        except asyncio.CancelledError:
            break
        except Exception as exc:  # pragma: no cover
            logger.error("Order sync loop error", error=str(exc))
        await asyncio.sleep(interval_seconds)