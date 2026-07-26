"""BotRunner — schedules and executes all enabled bots via APScheduler."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select

from app.utils.logging import logger


def _first_run_time() -> datetime:
    """
    Compute the first evaluation time for a bot.

    Returns
    -------
    datetime
        A UTC timestamp a random number of seconds (30‑150) after the current
        time. This staggers the initial execution of bots to avoid a stampede
        when the application starts.
    """
    return datetime.now(timezone.utc) + timedelta(seconds=random.uniform(30, 150))


if TYPE_CHECKING:
    from app.models.bot import Bot


# Map interval strings to APScheduler kwargs
_INTERVAL_MAP: Dict[str, Dict[str, Any]] = {
    "1m": {"minutes": 1},
    "5m": {"minutes": 5},
    "15m": {"minutes": 15},
    "30m": {"minutes": 30},
    "1h": {"hours": 1},
    "4h": {"hours": 4},
    "1d": {"hours": 24},
}


class BotRunner:
    """
    Loads all enabled bots from the database and schedules them on an APScheduler
    instance.

    Parameters
    ----------
    scheduler : AsyncIOScheduler
        The APScheduler instance used to schedule bot evaluation jobs.
    """

    def __init__(self, scheduler: AsyncIOScheduler) -> None:
        self._scheduler = scheduler

    async def start(self) -> None:
        """
        Load all enabled bots from the database and schedule each one.

        This method queries the `Bot` table for records where `is_enabled` is
        true and `is_archived` is false, then calls :meth:`reschedule` for each
        bot. Any exception during the process is logged.
        """
        try:
            from app.database import AsyncSessionLocal
            from app.models.bot import Bot

            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Bot).where(
                        Bot.is_enabled == True,  # noqa: E712
                        Bot.is_archived == False,  # noqa: E712
                    )
                )
                bots = result.scalars().all()

            logger.info("BotRunner: scheduling bots", count=len(bots))
            for bot in bots:
                await self.reschedule(bot)
        except Exception as exc:
            logger.error("BotRunner.start failed", error=str(exc))

    async def _run_bot(self, bot_id: str) -> None:
        """
        Execute a single bot evaluation.

        Parameters
        ----------
        bot_id : str
            The unique identifier of the bot to evaluate.

        The method retrieves the bot from the database, ensures it is enabled,
        and then runs the evaluation through ``BotEngine``. Results and any
        errors are logged.
        """
        try:
            from app.database import AsyncSessionLocal
            from app.models.bot import Bot
            from app.bots.engine import BotEngine

            engine = BotEngine()
            async with AsyncSessionLocal() as db:
                result = await db.execute(select(Bot).where(Bot.id == bot_id))
                bot = result.scalar_one_or_none()
                if bot is None or not bot.is_enabled:
                    return
                bot_result = await engine.evaluate(bot, db)
                logger.info(
                    "Bot evaluated",
                    bot_id=bot_id,
                    bot_name=bot.name,
                    fired=bot_result.fired,
                    signal=bot_result.signal,
                    reason=bot_result.reason,
                )
        except Exception as exc:
            logger.error("Bot run failed", bot_id=bot_id, error=str(exc))

    async def reschedule(self, bot: "Bot") -> None:
        """
        Add or update a bot job in the scheduler.

        Parameters
        ----------
        bot : Bot
            The bot model instance containing scheduling configuration.

        The method determines the appropriate trigger type and configures the
        APScheduler job accordingly, using a staggered first run time. Errors are
        logged but do not raise.
        """
        try:
            trigger_cfg: Dict[str, Any] = bot.trigger or {}
            trigger_type = trigger_cfg.get("type", "schedule")

            job_id = f"bot_{bot.id}"

            if trigger_type == "schedule":
                interval_str = trigger_cfg.get("interval", "1h")
                interval_kwargs = _INTERVAL_MAP.get(interval_str, {"hours": 1})
                self._scheduler.add_job(
                    self._run_bot,
                    "interval",
                    kwargs={"bot_id": bot.id},
                    id=job_id,
                    replace_existing=True,
                    max_instances=1,
                    next_run_time=_first_run_time(),
                    **interval_kwargs,
                )
                logger.debug("Bot scheduled", bot_id=bot.id, interval=interval_str)

            elif trigger_type in ("price_cross", "indicator"):
                # For non‑schedule triggers, poll every 5 minutes and let the engine decide
                self._scheduler.add_job(
                    self._run_bot,
                    "interval",
                    kwargs={"bot_id": bot.id},
                    id=job_id,
                    replace_existing=True,
                    max_instances=1,
                    next_run_time=_first_run_time(),
                    minutes=5,
                )
                logger.debug("Bot scheduled (poll)", bot_id=bot.id, trigger=trigger_type)

        except Exception as exc:
            logger.error("BotRunner.reschedule failed", bot_id=bot.id, error=str(exc))

    async def unschedule(self, bot_id: str) -> None:
        """
        Remove a bot job from the scheduler.

        Parameters
        ----------
        bot_id : str
            The identifier of the bot to unschedule.

        If the job does not exist, the exception is caught and logged at debug
        level.
        """
        job_id = f"bot_{bot_id}"
        try:
            self._scheduler.remove_job(job_id)
            logger.debug("Bot unscheduled", bot_id=bot_id)
        except Exception as exc:  # noqa: BLE001 — job may simply not exist
            logger.debug("Bot unschedule skipped", bot_id=bot_id, error=str(exc))