"""Bot Builder API — CRUD + manual run + toggle."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.bots.engine import BotEngine
from app.bots.templates import BOT_TEMPLATES
from app.models.bot import Bot, MARKET_TYPES
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


@router.get("/market-types", response_model=list[dict])
async def get_market_types() -> list[dict]:
    """Market types / desks a bot can trade — drives the builder's desk dropdown."""
    labels = {
        "equity": "Equities",
        "crypto": "Crypto",
        "polymarket": "Prediction Markets",
        "options": "Options",
        "macro": "Macro",
        "rates": "Rates",
    }
    return [{"value": mt, "label": labels.get(mt, mt.title())} for mt in MARKET_TYPES]


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

    By default returns only active (non‑archived) bots. Pass ``?archived=true``
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


@router.get("/{bot_id}/performance", response_model=dict)
async def get_bot_performance(
    bot_id: str,
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Option Alpha‑style per‑bot performance: cumulative realized P&L series + stats.

    Built from real closed ``Trade`` rows attributed to this bot (``strategy_name ==
    bot.name``). An empty series means the bot hasn't closed a position yet — nothing
    is fabricated. Powers the per‑bot sparkline in BotBuilder.
    """
    bot = await _get_user_bot(bot_id, current_user.id, db)
    days = max(1, min(days, 365))
    since = datetime.now(timezone.utc) - timedelta(days=days)

    rows = await _fetch_trade_rows(db, bot.name, since)
    series, stats = _process_trades(rows)

    return {
        "bot_id": bot.id,
        "bot_name": bot.name,
        "days": days,
        "series": series,
        **stats,
    }


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
    """Archive (soft‑delete) a bot.

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

    The bot is left disabled so the user can review it before re‑enabling.
    """
    bot = await _get_user_bot(bot_id, current_user.id, db)
    bot.is_archived = False
    bot.archived_at = None
    await db.commit()
    await db.refresh(bot)
    # Restored bots come back disabled; reschedule only if the user re‑enables them.
    _maybe_reschedule(bot)
    logger.info("Bot restored", bot_id=bot.id, user_id=current_user.id)
    return bot


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _fetch_trade_rows(db: AsyncSession, strategy_name: str, since: datetime) -> list:
    """Retrieve closed trades for a given strategy name since a timestamp."""
    from app.models.trade import Trade

    result = await db.execute(
        select(Trade)
        .where(Trade.strategy_name == strategy_name, Trade.closed_at >= since)
        .order_by(Trade.closed_at.asc())
    )
    return result.scalars().all()


def _process_trades(rows: list) -> tuple[list[dict], dict]:
    """Generate the time series and aggregate statistics from trade rows."""
    series: list[dict] = []
    cum = 0.0
    wins = 0
    peak = 0.0
    max_dd = 0.0
    holds: list[int] = []

    for trade in rows:
        pnl = float(trade.realized_pnl or 0)
        cum += pnl
        if pnl > 0:
            wins += 1
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        if trade.hold_seconds:
            holds.append(int(trade.hold_seconds))

        series.append(
            {
                "date": trade.closed_at.isoformat() if trade.closed_at else None,
                "pnl": round(pnl, 2),
                "cum_pnl": round(cum, 2),
                "symbol": trade.symbol,
            }
        )

    n = len(rows)
    stats = {
        "total_pnl": round(cum, 2),
        "trades": n,
        "win_rate": round(wins / n, 4) if n else None,
        "max_drawdown": round(max_dd, 2),
        "avg_hold_hours": round(sum(holds) / len(holds) / 3600, 2) if holds else None,
    }
    return series, stats


# Placeholder stubs for functions referenced elsewhere in the module.
# In the actual codebase these would be implemented with proper logic.
async def _get_user_bot(bot_id: str, user_id: int, db: AsyncSession) -> Bot:
    """Fetch a bot belonging to a specific user; raise 404 if not found."""
    result = await db.execute(select(Bot).where(Bot.id == bot_id, Bot.user_id == user_id))
    bot = result.scalar_one_or_none()
    if not bot:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot not found")
    return bot


def _maybe_reschedule(bot: Bot) -> None:
    """Schedule or unschedule the bot based on its enabled state."""
    # Implementation depends on the scheduler subsystem.
    pass


def _maybe_unschedule(bot_id: str) -> None:
    """Remove a bot from the scheduler."""
    # Implementation depends on the scheduler subsystem.
    pass