"""BotRunner — schedules and executes all enabled bots via APScheduler."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select

from app.utils.logging import logger

if TYPE_CHECKING:
    from app.models.bot import Bot

# Map interval strings to APScheduler kwargs
_INTERVAL_MAP: dict[str, dict] = {
    "1m": {"minutes": 1},
    "5m": {"minutes": 5},
    "15m": {"minutes": 15},
    "30m": {"minutes": 30},
    "1h": {"hours": 1},
    "4h": {"hours": 4},
    "1d": {"hours": 24},
}


class BotRunner:
    """Loads all enabled bots from DB and schedules them on APScheduler."""

    def __init__(self, scheduler: AsyncIOScheduler):
        self._scheduler = scheduler

    async def start(self) -> None:
        """Load all enabled bots from DB and schedule them."""
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

            # Guard against None or empty collection
            if not bots:
                logger.info("BotRunner: no enabled bots to schedule")
                return

            logger.info("BotRunner: scheduling bots", count=len(bots))
            for bot in bots:
                await self.reschedule(bot)
        except Exception as exc:
            logger.error("BotRunner.start failed", error=str(exc))

    async def _run_bot(self, bot_id: str) -> None:
        """Called by scheduler — fetch bot from DB, evaluate, update."""
        # Guard against invalid bot_id
        if not bot_id:
            logger.debug("BotRunner._run_bot called with empty bot_id; skipping")
            return
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
        """Add or update a bot job in the scheduler."""
        # Guard against None bot or missing identifiers
        if bot is None:
            logger.debug("BotRunner.reschedule called with None bot; skipping")
            return
        if not getattr(bot, "id", None):
            logger.debug("BotRunner.reschedule called with bot missing id; skipping")
            return
        try:
            trigger_cfg: dict = bot.trigger or {}
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
                    **interval_kwargs,
                )
                logger.debug("Bot scheduled", bot_id=bot.id, interval=interval_str)

            elif trigger_type in ("price_cross", "indicator"):
                # For non-schedule triggers, poll every 5 minutes and let the engine decide
                self._scheduler.add_job(
                    self._run_bot,
                    "interval",
                    kwargs={"bot_id": bot.id},
                    id=job_id,
                    replace_existing=True,
                    max_instances=1,
                    minutes=5,
                )
                logger.debug("Bot scheduled (poll)", bot_id=bot.id, trigger=trigger_type)

        except Exception as exc:
            logger.error("BotRunner.reschedule failed", bot_id=getattr(bot, "id", "unknown"), error=str(exc))

    async def unschedule(self, bot_id: str) -> None:
        """Remove a bot job from the scheduler."""
        # Guard against empty bot_id
        if not bot_id:
            logger.debug("BotRunner.unschedule called with empty bot_id; skipping")
            return
        job_id = f"bot_{bot_id}"
        try:
            self._scheduler.remove_job(job_id)
            logger.debug("Bot unscheduled", bot_id=bot_id)
        except Exception as exc:  # noqa: BLE001 — job may simply not exist
            logger.debug("Bot unschedule skipped", bot_id=bot_id, error=str(exc))