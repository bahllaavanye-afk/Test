"""APScheduler setup: hourly snapshots, nightly retraining, order sync."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel, Field, validator
from sqlalchemy import select

from app.utils.logging import logger

_scheduler: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler:
    """Return a singleton ``AsyncIOScheduler`` instance.

    The scheduler is created lazily on first call and configured to use UTC
    timezone. Subsequent calls return the same instance, ensuring that jobs are
    not duplicated.
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone="UTC")
    return _scheduler


class SchedulerJobConfig(BaseModel):
    """Configuration model for APScheduler jobs.

    This model is used internally to validate job parameters before they are
    added to the scheduler. It provides clear field descriptions, example values,
    and basic validation to prevent mis‑configuration.
    """

    job_id: str = Field(
        ...,
        description="Unique identifier for the job within the scheduler.",
        example="snapshot",
        min_length=1,
    )
    trigger: str = Field(
        ...,
        description="APScheduler trigger type (e.g., ``interval`` or ``cron``).",
        example="interval",
    )
    trigger_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed to the trigger (e.g., ``hours=1``).",
        example={"hours": 1},
    )
    max_instances: int = Field(
        1,
        description="Maximum number of concurrent instances of this job.",
        ge=1,
        example=1,
    )
    replace_existing: bool = Field(
        True,
        description="Whether to replace an existing job with the same ID.",
        example=True,
    )
    func: Any = Field(
        ...,
        description="Callable that will be executed when the job runs.",
    )
    func_args: List[Any] = Field(
        default_factory=list,
        description="Positional arguments passed to ``func``.",
        example=[],
    )
    func_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed to ``func``.",
        example={},
    )
    description: Optional[str] = Field(
        None,
        description="Human‑readable description of the job's purpose.",
        example="Capture hourly account snapshots.",
    )

    @validator("trigger")
    def _validate_trigger(cls, v: str) -> str:
        allowed = {"interval", "cron", "date", "calendar"}
        if v not in allowed:
            raise ValueError(f"trigger must be one of {allowed}, got {v!r}")
        return v

    @validator("max_instances")
    def _validate_max_instances(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_instances must be at least 1")
        return v


def _add_job(scheduler: AsyncIOScheduler, config: SchedulerJobConfig) -> None:
    """Add a job to the scheduler using a validated ``SchedulerJobConfig``."""
    scheduler.add_job(
        config.func,
        config.trigger,
        id=config.job_id,
        replace_existing=config.replace_existing,
        max_instances=config.max_instances,
        **config.trigger_args,
        args=config.func_args,
        kwargs=config.func_kwargs,
    )
    logger.info(
        "Scheduled job added",
        job_id=config.job_id,
        trigger=config.trigger,
        trigger_args=config.trigger_args,
    )


def start_scheduler(db_session_factory, broker=None) -> AsyncIOScheduler:
    """Configure and start the APScheduler with background tasks.

    Args:
        db_session_factory: Callable that returns an async SQLAlchemy session.
        broker: Optional broker implementation (currently unused).

    Returns:
        The configured ``AsyncIOScheduler`` instance.
    """
    scheduler = get_scheduler()

    async def _hourly_snapshot() -> None:
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

            snap_records: List[AccountSnapshot] = []
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

    async def _nightly_retrain() -> None:
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

    async def _order_sync() -> None:
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

            updates: List[tuple[str, Dict[str, Any]]] = []
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
                        updates.append(
                            (
                                order_row.id,
                                {
                                    "status": data.get("status", order_row.status),
                                    "filled_qty": float(data.get("filled_qty") or 0),
                                    "avg_fill_price": (
                                        float(data["filled_avg_price"])
                                        if data.get("filled_avg_price")
                                        else None
                                    ),
                                },
                            )
                        )
                except Exception as exc:
                    logger.debug(
                        "Order sync: failed to fetch order",
                        order_id=order_row.id,
                        error=str(exc),
                    )

            if updates:
                async with factory() as db:
                    for order_id, fields in updates:
                        result = await db.execute(select(Order).where(Order.id == order_id))
                        order = result.scalar_one_or_none()
                        if order:
                            for key, val in fields.items():
                                setattr(order, key, val)
                    await db.commit()
                logger.info("Order sync complete", updated=len(updates))

        except Exception as exc:
            logger.error("Order sync failed", error=str(exc))

    # Register jobs using the validated configuration model
    _add_job(
        scheduler,
        SchedulerJobConfig(
            job_id="snapshot",
            trigger="interval",
            trigger_args={"hours": 1},
            func=_hourly_snapshot,
            description="Capture hourly equity snapshots for active accounts.",
        ),
    )
    _add_job(
        scheduler,
        SchedulerJobConfig(
            job_id="retrain",
            trigger="cron",
            trigger_args={"hour": 2, "minute": 0},
            func=_nightly_retrain,
            description="Run nightly ML model retraining at 02:00 UTC.",
        ),
    )
    _add_job(
        scheduler,
        SchedulerJobConfig(
            job_id="order_sync",
            trigger="interval",
            trigger_args={"minutes": 1},
            func=_order_sync,
            description="Synchronize open broker orders with the database every minute.",
        ),
    )

    async def _slack_employee_report() -> None:
        """Post hourly employee status to Slack #engineering."""
        try:
            from app.notifications.slack import slack
            from app.main import app as _app

            algo = getattr(_app.state, "algo_agent", None)
            research = getattr(_app.state, "research_sci", None)

            # Build a simple status payload; actual implementation may vary.
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "algo_agent_running": bool(algo),
                "research_scientist_active": bool(research),
            }
            await slack.post_message(channel="#engineering", text=str(payload))
            logger.info("Slack employee report posted")
        except Exception as exc:
            logger.error("Slack employee report failed", error=str(exc))

    # Schedule the Slack reporting job
    _add_job(
        scheduler,
        SchedulerJobConfig(
            job_id="slack_report",
            trigger="interval",
            trigger_args={"hours": 1},
            func=_slack_employee_report,
            description="Post hourly employee status to Slack.",
        ),
    )

    # main.py calls start_scheduler() and stores the result without calling .start()
    # itself, so this MUST return a *running* scheduler. A rewrite dropped the start()
    # call, which registered jobs but never ran them (snapshot/retrain/order_sync/
    # slack_report were silently dead). Guard against double-start.
    if not scheduler.running:
        scheduler.start()
    return scheduler