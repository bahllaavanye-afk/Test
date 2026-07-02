"""APScheduler setup: hourly snapshots, nightly retraining, order sync."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel, Field, validator
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.logging import logger

# Lazy‑loaded imports to avoid circular dependencies
from app.models.account import Account, AccountSnapshot
from app.models.order import Order
from app.brokers.alpaca_orders import get_alpaca_account, _headers, _base_url
import httpx

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
    func: Callable[..., Any] = Field(
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


def start_scheduler(
    db_session_factory: Callable[[], AsyncSession],
    broker: Optional[Any] = None,
) -> AsyncIOScheduler:
    """Configure and start the APScheduler with background tasks.

    Args:
        db_session_factory: Callable that returns an async SQLAlchemy session.
        broker: Optional broker implementation (currently unused).

    Returns:
        The configured ``AsyncIOScheduler`` instance.
    """
    scheduler = get_scheduler()

    async def _hourly_snapshot() -> None:
        """Capture an equity snapshot for every active account."""
        logger.info("Running hourly account snapshot")
        factory = db_session_factory or (
            __import__("app.database", fromlist=["AsyncSessionLocal"]).AsyncSessionLocal
        )

        try:
            async with factory() as db:
                result = await db.execute(
                    select(Account).where(Account.is_active.is_(True))
                )
                accounts = result.scalars().all()
        except Exception as exc:
            logger.error("Failed to fetch active accounts", error=str(exc))
            return

        if not accounts:
            logger.info("Hourly snapshot: no active accounts")
            return

        snap_records: List[AccountSnapshot] = []
        for acct in accounts:
            if acct.broker != "alpaca" or not acct.encrypted_key:
                continue
            try:
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
            try:
                async with factory() as db:
                    db.add_all(snap_records)
                    await db.commit()
                logger.info("Hourly snapshot saved", count=len(snap_records))
            except Exception as exc:
                logger.error("Failed to persist snapshots", error=str(exc))
        else:
            logger.info("Hourly snapshot: no broker accounts with credentials")

    async def _nightly_retrain() -> None:
        """Trigger nightly ML model retraining at 02:00 UTC."""
        logger.info("Nightly ML retrain triggered")
        try:
            from app.tasks.ml_retrain import nightly_retrain

            await nightly_retrain()
        except Exception as exc:
            logger.error("Nightly retrain failed", error=str(exc))

    async def _order_sync() -> None:
        """Sync open broker orders back to the DB every minute."""
        logger.info("Order sync tick")
        factory = db_session_factory or (
            __import__("app.database", fromlist=["AsyncSessionLocal"]).AsyncSessionLocal
        )

        try:
            async with factory() as db:
                result = await db.execute(
                    select(Order).where(
                        and_(
                            Order.status.in_(["open", "partial"]),
                            Order.is_active.is_(True),
                        )
                    )
                )
                open_orders = result.scalars().all()
        except Exception as exc:
            logger.error("Failed to fetch open orders", error=str(exc))
            return

        if not open_orders:
            logger.debug("Order sync: no open orders")
            return

        # Reuse a single HTTP client for all broker calls
        async with httpx.AsyncClient(timeout=10.0) as client:
            updated_orders: List[Order] = []
            for order in open_orders:
                if order.broker != "alpaca" or not order.account_id:
                    continue
                try:
                    url = f"{_base_url}/orders/{order.broker_order_id}"
                    resp = await client.get(url, headers=_headers)
                    resp.raise_for_status()
                    data = resp.json()
                    # Update only fields that may have changed
                    order.status = data.get("status", order.status)
                    order.filled_qty = float(data.get("filled_qty", order.filled_qty))
                    order.avg_fill_price = float(
                        data.get("filled_avg_price", order.avg_fill_price)
                    )
                    updated_orders.append(order)
                except Exception as exc:
                    logger.warning(
                        "Order sync failed for order",
                        order_id=order.id,
                        broker=order.broker,
                        error=str(exc),
                    )

        if updated_orders:
            try:
                async with factory() as db:
                    await db.commit()
                logger.info("Order sync completed", updated=len(updated_orders))
            except Exception as exc:
                logger.error("Failed to commit order sync updates", error=str(exc))
        else:
            logger.debug("Order sync: no updates applied")

    # Register jobs with appropriate triggers
    _add_job(
        scheduler,
        SchedulerJobConfig(
            job_id="hourly_snapshot",
            trigger="interval",
            trigger_args={"hours": 1},
            func=_hourly_snapshot,
            description="Capture hourly account snapshots.",
        ),
    )
    _add_job(
        scheduler,
        SchedulerJobConfig(
            job_id="nightly_retrain",
            trigger="cron",
            trigger_args={"hour": 2, "minute": 0},
            func=_nightly_retrain,
            description="Trigger nightly ML model retraining.",
        ),
    )
    _add_job(
        scheduler,
        SchedulerJobConfig(
            job_id="order_sync",
            trigger="interval",
            trigger_args={"minutes": 1},
            func=_order_sync,
            description="Sync open broker orders to the DB.",
        ),
    )

    if not scheduler.running:
        scheduler.start()
        logger.info("Scheduler started")
    else:
        logger.debug("Scheduler already running")

    return scheduler