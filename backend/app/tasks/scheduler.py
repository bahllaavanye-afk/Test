"""APScheduler setup: hourly snapshots, nightly retraining, order sync."""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from app.utils.logging import logger

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_TIMEZONE: str = "UTC"

# Job identifiers
SNAPSHOT_JOB_ID: str = "hourly_snapshot"
NIGHTLY_RETRAIN_JOB_ID: str = "nightly_retrain"
ORDER_SYNC_JOB_ID: str = "order_sync"

# Triggers
INTERVAL_TRIGGER: str = "interval"
CRON_TRIGGER: str = "cron"

# Trigger arguments
SNAPSHOT_TRIGGER_ARGS: Dict[str, Any] = {"hours": 1}
NIGHTLY_RETRAIN_TRIGGER_ARGS: Dict[str, Any] = {"hour": 2, "minute": 0}
ORDER_SYNC_TRIGGER_ARGS: Dict[str, Any] = {"minutes": 1}

# Descriptions
SNAPSHOT_DESCRIPTION: str = "Capture hourly account snapshots."
NIGHTLY_RETRAIN_DESCRIPTION: str = "Trigger nightly ML model retraining."
ORDER_SYNC_DESCRIPTION: str = "Sync open broker orders."

# --------------------------------------------------------------------------- #
# Scheduler singleton
# --------------------------------------------------------------------------- #
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


class SchedulerJobConfig(BaseModel):
    """Configuration model for APScheduler jobs.

    This model is used internally to validate job parameters before they are
    added to the scheduler. It provides clear field descriptions, example values,
    and basic validation to prevent mis‑configuration.
    """

    job_id: str = Field(
        ...,
        description="Unique identifier for the job within the scheduler.",
        min_length=1,
        json_schema_extra={"example": "snapshot"},
    )
    trigger: str = Field(
        ...,
        description="APScheduler trigger type (e.g., ``interval`` or ``cron``).",
        json_schema_extra={"example": "interval"},
    )
    trigger_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed to the trigger (e.g., ``hours=1``).",
        json_schema_extra={"example": {"hours": 1}},
    )
    max_instances: int = Field(
        1,
        description="Maximum number of concurrent instances of this job.",
        ge=1,
        json_schema_extra={"example": 1},
    )
    replace_existing: bool = Field(
        True,
        description="Whether to replace an existing job with the same ID.",
        json_schema_extra={"example": True},
    )
    func: Any = Field(
        ...,
        description="Callable that will be executed when the job runs.",
    )
    func_args: List[Any] = Field(
        default_factory=list,
        description="Positional arguments passed to ``func``.",
        json_schema_extra={"example": []},
    )
    func_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed to ``func``.",
        json_schema_extra={"example": {}},
    )
    description: Optional[str] = Field(
        None,
        description="Human‑readable description of the job's purpose.",
        json_schema_extra={"example": "Capture hourly account snapshots."},
    )

    @field_validator("trigger")
    @classmethod
    def _validate_trigger(cls, v: str) -> str:
        allowed = {"interval", "cron", "date", "calendar"}
        if v not in allowed:
            raise ValueError(f"trigger must be one of {allowed}, got {v!r}")
        return v

    @field_validator("max_instances")
    @classmethod
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
            # Placeholder for actual sync logic; implementation details omitted.
            pass
        except Exception as exc:
            logger.error("Order sync failed", error=str(exc))

    # ----------------------------------------------------------------------- #
    # Register jobs using extracted constants
    # ----------------------------------------------------------------------- #
    _add_job(
        scheduler,
        SchedulerJobConfig(
            job_id=SNAPSHOT_JOB_ID,
            trigger=INTERVAL_TRIGGER,
            trigger_args=SNAPSHOT_TRIGGER_ARGS,
            func=_hourly_snapshot,
            description=SNAPSHOT_DESCRIPTION,
        ),
    )

    _add_job(
        scheduler,
        SchedulerJobConfig(
            job_id=NIGHTLY_RETRAIN_JOB_ID,
            trigger=CRON_TRIGGER,
            trigger_args=NIGHTLY_RETRAIN_TRIGGER_ARGS,
            func=_nightly_retrain,
            description=NIGHTLY_RETRAIN_DESCRIPTION,
        ),
    )

    _add_job(
        scheduler,
        SchedulerJobConfig(
            job_id=ORDER_SYNC_JOB_ID,
            trigger=INTERVAL_TRIGGER,
            trigger_args=ORDER_SYNC_TRIGGER_ARGS,
            func=_order_sync,
            description=ORDER_SYNC_DESCRIPTION,
        ),
    )

    # Start the scheduler if it is not already running.
    if not scheduler.running:
        scheduler.start()
        logger.info("Scheduler started")

    return scheduler


def shutdown_scheduler() -> None:
    """Gracefully shut down the APScheduler if it has been started."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=True)
        logger.info("Scheduler shut down")
        _scheduler = None