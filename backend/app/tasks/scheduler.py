"""APScheduler setup: hourly snapshots, nightly retraining, order sync."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel, Field, validator
from sqlalchemy import select

from app.utils.logging import logger

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

DEFAULT_TIMEZONE: str = "UTC"

# Job identifiers
JOB_ID_HOURLY_SNAPSHOT: str = "hourly_snapshot"
JOB_ID_NIGHTLY_RETRAIN: str = "nightly_retrain"
JOB_ID_ORDER_SYNC: str = "order_sync"

# Trigger types
TRIGGER_INTERVAL: str = "interval"
TRIGGER_CRON: str = "cron"

# Scheduling intervals / cron specifications
HOURLY_SNAPSHOT_INTERVAL_HOURS: int = 1
NIGHTLY_RETRAIN_CRON_HOUR: int = 2
NIGHTLY_RETRAIN_CRON_MINUTE: int = 0
ORDER_SYNC_INTERVAL_MINUTES: int = 1

# Broker identifiers
BROKER_ALPACA: str = "alpaca"

# -------------------------------------------------------------------------
# Scheduler singleton
# -------------------------------------------------------------------------

_scheduler: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler:
    """Return a singleton ``AsyncIOScheduler`` instance.

    The scheduler is created lazily on first call and configured to use UTC
    timezone. Subsequent calls return the same instance, ensuring that jobs are
    not duplicated.
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone=DEFAULT_TIMEZONE)
    return _scheduler


# -------------------------------------------------------------------------
# Pydantic model for job configuration
# -------------------------------------------------------------------------

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
        allowed = {TRIGGER_INTERVAL, TRIGGER_CRON, "date", "calendar"}
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


# -------------------------------------------------------------------------
# Scheduler tasks
# -------------------------------------------------------------------------

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
                    if acct.broker == BROKER_ALPACA and acct.encrypted_key:
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
                    select(Order).where(Order.status.in_(["open", "partial"]))
                )
                open_orders = result.scalars().all()

            # Fetch corresponding account credentials
            async with factory() as db:
                result = await db.execute(
                    select(Account).where(Account.id.in_([o.account_id for o in open_orders]))
                )
                accounts = {a.id: a for a in result.scalars().all()}

            async with httpx.AsyncClient() as client:
                for order in open_orders:
                    account = accounts.get(order.account_id)
                    if not account or account.broker != BROKER_ALPACA or not account.encrypted_key:
                        continue

                    url = f"{_base_url}/orders/{order.broker_order_id}"
                    try:
                        resp = await client.get(url, headers=_headers(account))
                        resp.raise_for_status()
                        data = resp.json()
                        order.status = data.get("status", order.status)
                        order.filled_qty = float(data.get("filled_qty", order.filled_qty))
                        order.avg_fill_price = float(data.get("filled_avg_price", order.avg_fill_price))
                    except Exception as exc:
                        logger.warning(
                            "Order sync failed",
                            order_id=order.id,
                            error=str(exc),
                        )

            # Persist updates
            async with factory() as db:
                db.add_all(open_orders)
                await db.commit()
            logger.info("Order sync completed", count=len(open_orders))

        except Exception as exc:
            logger.error("Order sync failed", error=str(exc))

    # ---------------------------------------------------------------------
    # Register jobs
    # ---------------------------------------------------------------------

    # Hourly snapshot (interval trigger)
    _add_job(
        scheduler,
        SchedulerJobConfig(
            job_id=JOB_ID_HOURLY_SNAPSHOT,
            trigger=TRIGGER_INTERVAL,
            trigger_args={"hours": HOURLY_SNAPSHOT_INTERVAL_HOURS},
            func=_hourly_snapshot,
            description="Capture hourly account snapshots.",
        ),
    )

    # Nightly retrain (cron trigger)
    _add_job(
        scheduler,
        SchedulerJobConfig(
            job_id=JOB_ID_NIGHTLY_RETRAIN,
            trigger=TRIGGER_CRON,
            trigger_args={"hour": NIGHTLY_RETRAIN_CRON_HOUR, "minute": NIGHTLY_RETRAIN_CRON_MINUTE},
            func=_nightly_retrain,
            description="Trigger nightly ML model retraining.",
        ),
    )

    # Order sync (interval trigger)
    _add_job(
        scheduler,
        SchedulerJobConfig(
            job_id=JOB_ID_ORDER_SYNC,
            trigger=TRIGGER_INTERVAL,
            trigger_args={"minutes": ORDER_SYNC_INTERVAL_MINUTES},
            func=_order_sync,
            description="Sync open broker orders to the database.",
        ),
    )

    # Start the scheduler if not already running
    if not scheduler.running:
        scheduler.start()
        logger.info("APScheduler started")

    return scheduler