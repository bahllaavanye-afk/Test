"""
Order sync task: polls broker every 15 seconds, detects new fills,
broadcasts via WebSocket, and posts to AgentBus.
"""
from __future__ import annotations

import asyncio

from app.utils.logging import logger


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

    try:
        import httpx
    except ImportError:
        logger.warning("httpx not available — order sync skipped")
        return

    try:
        async with db_session_factory() as db:
            result = await db.execute(
                select(Order, Account)
                .join(Account, Order.account_id == Account.id)
                .where(
                    Order.status.in_(["pending", "accepted", "partially_filled", "new"]),
                    Account.is_active == True,  # noqa: E712
                )
            )
            rows = result.all()
    except Exception as exc:
        logger.debug("Order sync: DB fetch failed", error=str(exc))
        return

    if not rows:
        return

    updates: list[tuple] = []  # (order_id, fields, user_id, original_order, new_status, new_filled_qty)
    for order_row, acct in rows:
        try:
            if not order_row.broker_order_id or acct.broker != "alpaca":
                continue
            headers = await _headers(acct)
            base = _base_url(acct)
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(
                    f"{base}/v2/orders/{order_row.broker_order_id}",
                    headers=headers,
                )
            if resp.status_code == 200:
                data = resp.json()
                new_status = data.get("status", order_row.status)
                new_filled_qty = float(data.get("filled_qty") or 0)
                new_avg_price = (
                    float(data["filled_avg_price"])
                    if data.get("filled_avg_price") else None
                )
                updates.append((
                    order_row.id,
                    {
                        "status": new_status,
                        "filled_qty": new_filled_qty,
                        "avg_fill_price": new_avg_price,
                    },
                    getattr(acct, "user_id", None),
                    order_row,
                    new_status,
                    new_filled_qty,
                ))
        except Exception as exc:
            logger.debug(
                "Order sync: failed to fetch order",
                order_id=order_row.id,
                error=str(exc),
            )

    if not updates:
        return

    new_fills: list[tuple] = []

    try:
        async with db_session_factory() as db:
            for entry in updates:
                order_id, fields, user_id, original_order, new_status, new_filled_qty = entry
                result = await db.execute(
                    select(Order).where(Order.id == order_id)
                )
                order = result.scalar_one_or_none()
                if order:
                    prev_filled_qty = float(order.filled_qty or 0)
                    prev_status = order.status
                    for key, val in fields.items():
                        setattr(order, key, val)

                    # Detect new fill: status became 'filled' or filled_qty increased
                    is_new_fill = (
                        (new_status == "filled" and prev_status != "filled")
                        or (new_filled_qty > prev_filled_qty and new_filled_qty > 0)
                    )
                    if is_new_fill:
                        new_fills.append((order, user_id))
            await db.commit()
        logger.info("Order sync complete", updated=len(updates))
    except Exception as exc:
        logger.error("Order sync DB update failed", error=str(exc))
        return

    # ── Broadcast fills via WebSocket and AgentBus ───────────────────────────
    for order, user_id in new_fills:
        try:
            uid = str(user_id) if user_id else "system"
            await manager.broadcast(f"orders:{uid}", {
                "type": "fill",
                "order_id": str(order.id),
                "symbol": order.symbol,
                "filled_qty": float(order.filled_qty or 0),
                "filled_avg_price": float(order.avg_fill_price or 0),
            })
            await manager.broadcast(f"positions:{uid}", {
                "type": "update",
                "symbol": order.symbol,
            })
        except Exception as exc:
            logger.debug("Order sync: WS broadcast failed", order_id=order.id, error=str(exc))

        try:
            from app.tasks.agent_bus import get_bus
            bus = get_bus()
            await bus.post_finding(
                "risk",
                f"Fill: {order.symbol} {order.filled_qty} @ {order.avg_fill_price}",
                {"order_id": str(order.id)},
                from_agent="order_sync",
            )
        except Exception as exc:
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
        except Exception as exc:
            logger.error("Order sync loop error", error=str(exc))
        await asyncio.sleep(interval_seconds)
