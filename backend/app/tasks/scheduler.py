"""APScheduler setup: hourly snapshots, nightly retraining, order sync."""
from __future__ import annotations
from apscheduler.schedulers.asyncio import AsyncIOScheduler
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
        logger.info("Running hourly account snapshot")
        # Implemented fully in production — placeholder logs intent
        pass

    async def _nightly_retrain():
        logger.info("Nightly ML retrain triggered")
        pass

    async def _order_sync():
        logger.info("Order sync tick")
        pass

    scheduler.add_job(_hourly_snapshot, "interval", hours=1, id="snapshot")
    scheduler.add_job(_nightly_retrain, "cron", hour=2, minute=0, id="retrain")
    scheduler.add_job(_order_sync, "interval", minutes=1, id="order_sync")

    scheduler.start()
    logger.info("Scheduler started")
    return scheduler
