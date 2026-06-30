"""BotRunner — schedules and executes all enabled bots via APScheduler."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from apscheduler.jobstores.base import JobLookupError

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
        # Import dependencies; handle import errors explicitly
        try:
            from app.database import AsyncSessionLocal
            from app.models.bot import Bot
        except ImportError as exc:
            logger.error("BotRunner.start import failed", error=str(exc))
            return

        # Retrieve bots; handle DB‑related errors separately
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Bot).where(
                        Bot.is_enabled == True,  # noqa: E712
                        Bot.is_archived == False,  # noqa: E712
                    )
                )
                bots = result.scalars().all()
        except SQLAlchemyError as exc:
            logger.exception("BotRunner.start DB query failed", error=str(exc))
            return
        except Exception as exc:
            logger.exception("BotRunner.start unexpected error", error=str(exc))
            return

        logger.info("BotRunner: scheduling bots", count=len(bots))
        for bot in bots:
            try:
                await self.reschedule(bot)
            except Exception as exc:
                logger.exception(
                    "BotRunner.start failed to reschedule bot",
                    bot_id=bot.id,
                    error=str(exc),
                )

    async def _run_bot(self, bot_id: str) -> None:
        """Called by scheduler — fetch bot from DB, evaluate, update."""
        try:
            from app.database import AsyncSessionLocal
            from app.models.bot import Bot
            from app.bots.engine import BotEngine
        except ImportError as exc:
            logger.error("BotRunner._run_bot import failed", bot_id=bot_id, error=str(exc))
            return

        engine = BotEngine()
        try:
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
        except SQLAlchemyError as exc:
            logger.exception("BotRunner._run_bot DB error", bot_id=bot_id, error=str(exc))
        except Exception as exc:
            logger.exception("Bot run failed", bot_id=bot_id, error=str(exc))

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
                # For non‑schedule triggers, poll every 5 minutes and let the engine decide
                self._scheduler.add_job(
                    self._run_bot,
                    "interval",
                    kwargs={"bot_id": bot.id},
                    id=job_id,
                    replace_existing=True,
                    max_instances=1,
                    minutes=5,
                )
                logger.debug(
                    "Bot scheduled (poll)", bot_id=bot.id, trigger=trigger_type
                )
        except ValueError as exc:
            logger.error(
                "BotRunner.reschedule invalid schedule configuration",
                bot_id=bot.id,
                error=str(exc),
            )
        except Exception as exc:
            logger.exception(
                "BotRunner.reschedule failed", bot_id=bot.id, error=str(exc)
            )

    async def unschedule(self, bot_id: str) -> None:
        """Remove a bot job from the scheduler."""
        job_id = f"bot_{bot_id}"
        try:
            self._scheduler.remove_job(job_id)
            logger.debug("Bot unscheduled", bot_id=bot_id)
        except JobLookupError:
            logger.debug("Bot unschedule called but job not found", bot_id=bot_id)
        except Exception as exc:
            logger.exception(
                "BotRunner.unschedule unexpected error", bot_id=bot_id, error=str(exc)
            )