"""BotRunner — schedules and executes all enabled bots via APScheduler."""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select

from app.utils.logging import logger


def _first_run_time() -> datetime:
    """First evaluation shortly after boot (staggered 30–150s so 61 bots don't
    stampede). Without this, APScheduler interval jobs wait one FULL interval
    before the first run — and on the ephemeral-DB deploy, app restarts on every
    merge reset that clock, so 1h/1d bots NEVER got to run (every bot showed
    last_run_at=None). Found 2026-07-21 diagnosing 'OA doesn't work / 0 trades'.
    """
    return datetime.now(timezone.utc) + timedelta(seconds=random.uniform(30, 150))


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

    def _has_job(self, bot_id: str) -> bool:
        return self._scheduler.get_job(f"bot_{bot_id}") is not None

    async def _fetch_enabled_bots(self) -> List["Bot"]:
        """Retrieve all enabled and non‑archived bots from the database."""
        from app.database import AsyncSessionLocal
        from app.models.bot import Bot

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Bot).where(
                    Bot.is_enabled == True,  # noqa: E712
                    Bot.is_archived == False,  # noqa: E712
                )
            )
            return result.scalars().all()

    def _log_unscheduled(self, bots: List["Bot"]) -> None:
        """Log an error for any enabled bot that lacks a scheduler job."""
        unscheduled = [b.id for b in bots if not self._has_job(b.id)]
        if unscheduled:
            logger.error(
                "BotRunner: %d enabled bot(s) have NO scheduler job — they "
                "cannot fire and no orders will be placed",
                len(unscheduled),
                enabled=len(bots),
                unscheduled=len(unscheduled),
            )

    async def start(self, only_missing: bool = False) -> int:
        """Load all enabled bots from DB and schedule them. Returns how many.

        `only_missing` skips bots that already hold a scheduler job. That is
        what makes this safe to re-run: a blanket reschedule would reset every
        bot's `next_run_time` to `_first_run_time()` on every pass, so a bot on
        a short interval would have its next run pushed back indefinitely and
        never fire.

        Ignition used to be a ONE-SHOT job at boot. On the ephemeral SQLite
        fallback the bots table is empty at that moment, so it scheduled zero
        bots, logged `count=0`, and never looked again — the 61 bots seeded
        afterwards sat enabled-but-unscheduled forever. Observed live:
        61 enabled bots, `jobs_total=11`, `bot_jobs=2` (the exit-checker and
        lifecycle jobs), every bot at `last_run_at=None`, zero orders, zero
        trades.
        """
        try:
            bots = await self._fetch_enabled_bots()
            pending = [b for b in bots if not (only_missing and self._has_job(b.id))]
            logger.info(
                "BotRunner: scheduling bots",
                enabled=len(bots),
                scheduling=len(pending),
                only_missing=only_missing,
            )
            for bot in pending:
                await self.reschedule(bot)

            self._log_unscheduled(bots)
            return len(pending)
        except Exception as exc:
            logger.error("BotRunner.start failed", error=str(exc), exc_info=exc)
            return 0

    async def _run_bot(self, bot_id: str) -> None:
        """Called by scheduler — fetch bot from DB, evaluate, update."""
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
                    next_run_time=_first_run_time(),
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
                    next_run_time=_first_run_time(),
                    minutes=5,
                )
                logger.debug(
                    "Bot scheduled (poll)", bot_id=bot.id, trigger=trigger_type
                )

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