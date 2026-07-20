"""BotRunner — schedules and executes all enabled bots via APScheduler."""
from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Dict, Tuple

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

    # Cache lifetime in seconds
    _CACHE_TTL = 60

    def __init__(self, scheduler: AsyncIOScheduler):
        self._scheduler = scheduler
        # bot_id -> (Bot instance, timestamp)
        self._bot_cache: Dict[str, Tuple["Bot", float]] = {}

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

            logger.info("BotRunner: scheduling bots", count=len(bots))
            for bot in bots:
                # Prime cache to avoid immediate DB hit on first run
                self._bot_cache[bot.id] = (bot, time.time())
                await self.reschedule(bot)
        except Exception as exc:
            logger.error("BotRunner.start failed", error=str(exc))

    async def _get_bot_cached(self, bot_id: str, db) -> "Bot | None":
        """Retrieve bot from cache if fresh; otherwise query DB and update cache."""
        cached = self._bot_cache.get(bot_id)
        now = time.time()
        if cached:
            bot_obj, ts = cached
            if now - ts < self._CACHE_TTL:
                return bot_obj

        result = await db.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()
        if bot:
            self._bot_cache[bot_id] = (bot, now)
        return bot

    async def _run_bot(self, bot_id: str) -> None:
        """Called by scheduler — fetch bot from DB, evaluate, update."""
        try:
            from app.database import AsyncSessionLocal
            from app.models.bot import Bot
            from app.bots.engine import BotEngine

            engine = BotEngine()
            async with AsyncSessionLocal() as db:
                bot = await self._get_bot_cached(bot_id, db)
                if bot is None or not bot.is_enabled:
                    # Ensure stale entries are removed from cache
                    self._bot_cache.pop(bot_id, None)
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
            logger.error("BotRunner.reschedule failed", bot_id=bot.id, error=str(exc))

    async def unschedule(self, bot_id: str) -> None:
        """Remove a bot job from the scheduler."""
        job_id = f"bot_{bot_id}"
        try:
            self._scheduler.remove_job(job_id)
            logger.debug("Bot unscheduled", bot_id=bot_id)
        except Exception as exc:  # noqa: BLE001 — job may simply not exist
            logger.debug("Bot unschedule skipped", bot_id=bot_id, error=str(exc))