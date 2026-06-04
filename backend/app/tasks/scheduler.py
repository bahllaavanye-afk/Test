"""APScheduler setup: hourly snapshots, nightly retraining, order sync."""
from __future__ import annotations
import asyncio
import uuid
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select

from app.utils.logging import logger

_scheduler: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone="UTC")
    return _scheduler


def start_scheduler(db_session_factory, broker=None) -> AsyncIOScheduler:
    scheduler = get_scheduler()

    async def _hourly_snapshot():
        """
        Capture an equity snapshot for every active account.
        Fetches live account data from the broker and persists an AccountSnapshot row.
        """
        logger.info("Running hourly account snapshot")
        if db_session_factory is None:
            # Fallback: create a fresh session from the global engine
            try:
                from app.database import AsyncSessionLocal as _factory
                factory = _factory
            except Exception as exc:
                logger.warning("Snapshot: no DB session factory", error=str(exc))
                return
        else:
            factory = db_session_factory

        try:
            from app.models.account import Account, AccountSnapshot
            from app.brokers.alpaca_orders import get_alpaca_account

            async with factory() as db:
                result = await db.execute(
                    select(Account).where(Account.is_active == True)  # noqa: E712
                )
                accounts = result.scalars().all()

            snap_records: list[AccountSnapshot] = []
            for acct in accounts:
                try:
                    if acct.broker == "alpaca" and acct.encrypted_key:
                        data = await get_alpaca_account(acct)
                        snap = AccountSnapshot(
                            id=str(uuid.uuid4()),
                            account_id=acct.id,
                            ts=datetime.now(timezone.utc),
                            total_equity=float(data.get("equity", 0)),
                            cash=float(data.get("cash", 0)),
                            unrealized_pnl=float(data.get("unrealized_pl", 0)),
                            raw_payload=data,
                        )
                        snap_records.append(snap)
                except Exception as exc:
                    logger.warning(
                        "Snapshot fetch failed",
                        account_id=acct.id,
                        broker=acct.broker,
                        error=str(exc),
                    )

            if snap_records:
                async with factory() as db:
                    db.add_all(snap_records)
                    await db.commit()
                logger.info("Hourly snapshot saved", count=len(snap_records))
            else:
                logger.info("Hourly snapshot: no active broker accounts with credentials")

        except Exception as exc:
            logger.error("Hourly snapshot failed", error=str(exc))

    async def _nightly_retrain():
        """
        Trigger nightly ML model retraining at 02:00 UTC.
        Delegates to ml_retrain.nightly_retrain() which downloads data and trains.
        """
        logger.info("Nightly ML retrain triggered")
        try:
            from app.tasks.ml_retrain import nightly_retrain
            await nightly_retrain()
        except Exception as exc:
            logger.error("Nightly retrain failed", error=str(exc))

    async def _order_sync():
        """
        Sync open broker orders back to the DB every minute.
        Updates status, filled_qty, and avg_fill_price for pending/partial fills.
        """
        logger.info("Order sync tick")
        if db_session_factory is None:
            try:
                from app.database import AsyncSessionLocal as _factory
                factory = _factory
            except Exception as exc:
                logger.debug("Order sync: no DB session factory", error=str(exc))
                return
        else:
            factory = db_session_factory

        try:
            from app.models.order import Order
            from app.models.account import Account
            from app.brokers.alpaca_orders import _headers, _base_url
            import httpx

            # Fetch all open orders from the DB
            async with factory() as db:
                result = await db.execute(
                    select(Order, Account)
                    .join(Account, Order.account_id == Account.id)
                    .where(
                        Order.status.in_(["pending", "accepted", "partially_filled", "new"]),
                        Account.is_active == True,  # noqa: E712
                    )
                )
                rows = result.all()

            if not rows:
                return

            updates: list[tuple[str, dict]] = []
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
                        updates.append((order_row.id, {
                            "status": data.get("status", order_row.status),
                            "filled_qty": float(data.get("filled_qty") or 0),
                            "avg_fill_price": (
                                float(data["filled_avg_price"])
                                if data.get("filled_avg_price") else None
                            ),
                        }))
                except Exception as exc:
                    logger.debug(
                        "Order sync: failed to fetch order",
                        order_id=order_row.id,
                        error=str(exc),
                    )

            if updates:
                async with factory() as db:
                    for order_id, fields in updates:
                        result = await db.execute(
                            select(Order).where(Order.id == order_id)
                        )
                        order = result.scalar_one_or_none()
                        if order:
                            for key, val in fields.items():
                                setattr(order, key, val)
                    await db.commit()
                logger.info("Order sync complete", updated=len(updates))

        except Exception as exc:
            logger.error("Order sync failed", error=str(exc))

    scheduler.add_job(
        _hourly_snapshot,
        "interval",
        hours=1,
        id="snapshot",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        _nightly_retrain,
        "cron",
        hour=2,
        minute=0,
        id="retrain",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        _order_sync,
        "interval",
        minutes=1,
        id="order_sync",
        replace_existing=True,
        max_instances=1,
    )

    async def _slack_employee_report():
        """Post hourly employee status to Slack #engineering."""
        try:
            from app.notifications.slack import slack
            from app.main import app as _app
            from datetime import datetime, timezone

            algo = getattr(_app.state, "algo_agent", None)
            research = getattr(_app.state, "research_scientist", None)
            modeling = getattr(_app.state, "modeling_engineer", None)

            lines = [f"*QuantEdge Hourly Status* — {datetime.now(timezone.utc).strftime('%H:%M UTC')}"]
            if algo:
                lb = algo.get_leaderboard()
                best = lb[0] if lb else {}
                lines.append(f"• AlgoAgent: {algo._total_runs} runs | top: {best.get('strategy','?')} sharpe={best.get('avg_sharpe',0):.3f}")
            if research:
                s = research.get_research_summary()
                lines.append(f"• Research: {s.get('cycles_completed',0)} cycles | {s.get('total_findings',0)} findings | queue: {len(s.get('implement_queue',[]))} ideas")
            if modeling:
                e = modeling.get_engineering_summary()
                lines.append(f"• Modeling: {e.get('promote_count',0)} promotions | {e.get('retrain_count',0)} retrains")

            await slack.send("system", "system", "📊 Hourly Status", text="\n".join(lines))
        except Exception as exc:
            logger.debug("Slack employee report failed", error=str(exc))

    scheduler.add_job(
        _slack_employee_report,
        "interval",
        hours=1,
        id="slack_employee_report",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.start()
    logger.info("Scheduler started")
    return scheduler
