"""Bot Builder API — CRUD + manual run + toggle."""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.bots.engine import BotEngine
from app.bots.templates import BOT_TEMPLATES
from app.models.bot import Bot
from app.models.user import User
from app.schemas.bot import BotCreate, BotOut, BotUpdate
from app.utils.logging import logger

router = APIRouter(prefix="/bots", tags=["bots"])


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

@router.get("/templates", response_model=dict)
async def get_templates() -> dict:
    """Return all pre-built bot templates (no auth required)."""
    return BOT_TEMPLATES


# ---------------------------------------------------------------------------
# Protected CRUD
# ---------------------------------------------------------------------------

@router.get("/", response_model=list[BotOut])
async def list_bots(
    archived: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Bot]:
    """List bots belonging to the current user.

    By default returns only active (non-archived) bots. Pass ``?archived=true``
    to list archived bots instead.
    """
    result = await db.execute(
        select(Bot)
        .where(Bot.user_id == current_user.id, Bot.is_archived == archived)
        .order_by(Bot.created_at.desc())
    )
    return result.scalars().all()


@router.post("/", response_model=BotOut, status_code=status.HTTP_201_CREATED)
async def create_bot(
    payload: BotCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Bot:
    """Create a new bot."""
    bot = Bot(
        user_id=current_user.id,
        name=payload.name,
        description=payload.description,
        symbol=payload.symbol.upper(),
        market_type=payload.market_type,
        trigger=payload.trigger.model_dump(),
        conditions=[c.model_dump() for c in payload.conditions],
        condition_logic=payload.condition_logic,
        action=payload.action.model_dump(),
        exit_rules=[e.model_dump() for e in payload.exit_rules],
        template_id=payload.template_id,
    )
    db.add(bot)
    await db.commit()
    await db.refresh(bot)
    logger.info("Bot created", bot_id=bot.id, user_id=current_user.id, name=bot.name)
    return bot


@router.get("/{bot_id}", response_model=BotOut)
async def get_bot(
    bot_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Bot:
    """Get a single bot by ID."""
    bot = await _get_user_bot(bot_id, current_user.id, db)
    return bot


@router.patch("/{bot_id}", response_model=BotOut)
async def update_bot(
    bot_id: str,
    payload: BotUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Bot:
    """Update bot fields."""
    bot = await _get_user_bot(bot_id, current_user.id, db)

    if payload.name is not None:
        bot.name = payload.name
    if payload.description is not None:
        bot.description = payload.description
    if payload.is_enabled is not None:
        bot.is_enabled = payload.is_enabled
    if payload.conditions is not None:
        bot.conditions = [c.model_dump() for c in payload.conditions]
    if payload.condition_logic is not None:
        bot.condition_logic = payload.condition_logic
    if payload.action is not None:
        bot.action = payload.action.model_dump()
    if payload.exit_rules is not None:
        bot.exit_rules = [e.model_dump() for e in payload.exit_rules]

    await db.commit()
    await db.refresh(bot)

    # Reschedule if enabled/disabled changed
    _maybe_reschedule(bot)

    return bot


@router.delete("/{bot_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_bot(
    bot_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Archive (soft-delete) a bot.

    The row, its configuration, and any linked trades are preserved — the bot is
    simply marked archived, disabled, and removed from the scheduler. Use
    ``POST /bots/{id}/restore`` to bring it back. This replaces the old hard delete
    so bot history and performance are never lost.
    """
    bot = await _get_user_bot(bot_id, current_user.id, db)
    bot.is_archived = True
    bot.archived_at = datetime.now(UTC)
    bot.is_enabled = False
    await db.commit()
    _maybe_unschedule(bot_id)
    logger.info("Bot archived", bot_id=bot_id, user_id=current_user.id)


@router.post("/{bot_id}/restore", response_model=BotOut)
async def restore_bot(
    bot_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Bot:
    """Restore an archived bot back to the active list.

    The bot is left disabled so the user can review it before re-enabling.
    """
    bot = await _get_user_bot(bot_id, current_user.id, db)
    bot.is_archived = False
    bot.archived_at = None
    await db.commit()
    await db.refresh(bot)
    # Restored bots come back disabled; reschedule only if the user re-enables them.
    _maybe_reschedule(bot)
    logger.info("Bot restored", bot_id=bot_id, user_id=current_user.id)
    return bot


@router.post("/{bot_id}/run", response_model=dict)
async def run_bot(
    bot_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Manually trigger a bot evaluation right now."""
    bot = await _get_user_bot(bot_id, current_user.id, db)
    engine = BotEngine()
    result = await engine.evaluate(bot, db)
    return {
        "fired": result.fired,
        "reason": result.reason,
        "signal": result.signal,
        "orders_created": result.orders_created,
        "details": result.details,
    }


@router.post("/{bot_id}/toggle", response_model=BotOut)
async def toggle_bot(
    bot_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Bot:
    """Toggle bot enabled/disabled."""
    bot = await _get_user_bot(bot_id, current_user.id, db)
    bot.is_enabled = not bot.is_enabled
    await db.commit()
    await db.refresh(bot)
    _maybe_reschedule(bot)
    logger.info("Bot toggled", bot_id=bot_id, is_enabled=bot.is_enabled)
    return bot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_user_bot(bot_id: str, user_id: str, db: AsyncSession) -> Bot:
    """Fetch a bot and verify ownership."""
    result = await db.execute(select(Bot).where(Bot.id == bot_id, Bot.user_id == user_id))
    bot = result.scalar_one_or_none()
    if bot is None:
        raise HTTPException(status_code=404, detail="Bot not found")
    return bot


def _maybe_reschedule(bot: Bot) -> None:
    """Attempt to reschedule the bot in the global scheduler if available."""
    try:
        import asyncio
        from app.main import app as _app
        runner = getattr(_app.state, "bot_runner", None)
        if runner is None:
            return
        loop = asyncio.get_running_loop()
        if bot.is_enabled:
            loop.create_task(runner.reschedule(bot))
        else:
            loop.create_task(runner.unschedule(bot.id))
    except Exception as exc:
        logger.debug("Could not reschedule bot", bot_id=bot.id, error=str(exc))


def _maybe_unschedule(bot_id: str) -> None:
    """Attempt to unschedule a deleted bot."""
    try:
        import asyncio
        from app.main import app as _app
        runner = getattr(_app.state, "bot_runner", None)
        if runner is None:
            return
        loop = asyncio.get_running_loop()
        loop.create_task(runner.unschedule(bot_id))
    except Exception as exc:
        logger.debug("Could not unschedule bot", bot_id=bot_id, error=str(exc))
