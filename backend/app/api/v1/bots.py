"""Bot Builder API — CRUD + manual run + toggle + analytics."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.bots.engine import BotEngine
from app.bots.templates import BOT_TEMPLATES
from app.models.bot import Bot
from app.models.order import Order
from app.models.trade import Trade
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Bot]:
    """List all bots belonging to the current user."""
    result = await db.execute(
        select(Bot).where(Bot.user_id == current_user.id).order_by(Bot.created_at.desc())
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
        desk=payload.desk,
        signal_source=payload.signal_source,
        ml_model_name=payload.ml_model_name,
        ml_confidence_threshold=payload.ml_confidence_threshold,
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


@router.get("/summary/all", response_model=list[dict])
async def get_bots_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """
    Options Alpha command center: all bots with open positions count and
    30-day P&L. One row per bot — the 'bots dashboard' view.
    Registered before /{bot_id} so FastAPI doesn't capture 'summary' as a bot_id.
    """
    bots_result = await db.execute(
        select(Bot).where(Bot.user_id == current_user.id).order_by(Bot.created_at.desc())
    )
    bots = bots_result.scalars().all()

    since = datetime.now(UTC) - timedelta(days=30)
    rows = []
    for bot in bots:
        # Open positions
        pos_result = await db.execute(
            select(func.count()).select_from(Order).where(
                Order.status == "paper",
                Order.raw_payload["bot_id"].as_string() == bot.id,
            )
        )
        open_positions = pos_result.scalar() or 0

        # 30-day P&L
        trade_result = await db.execute(
            select(
                func.count().label("n"),
                func.sum(Trade.realized_pnl).label("total_pnl"),
            ).where(
                Trade.strategy_name == bot.name,
                Trade.closed_at >= since,
            )
        )
        row = trade_result.one()
        n_trades = row.n or 0
        total_pnl = float(row.total_pnl or 0)

        rows.append({
            "id": bot.id,
            "name": bot.name,
            "description": bot.description,
            "symbol": bot.symbol,
            "market_type": bot.market_type,
            "desk": bot.desk or "equity",
            "signal_source": bot.signal_source or "rule_based",
            "ml_model_name": bot.ml_model_name,
            "ml_confidence_threshold": float(bot.ml_confidence_threshold) if bot.ml_confidence_threshold else None,
            "is_enabled": bot.is_enabled,
            "run_count": bot.run_count,
            "last_run_at": bot.last_run_at.isoformat() if bot.last_run_at else None,
            "last_signal": bot.last_signal,
            "last_result": bot.last_result,
            "open_positions": open_positions,
            "trades_30d": n_trades,
            "pnl_30d": round(total_pnl, 4),
            "trigger_interval": (bot.trigger or {}).get("interval", "?"),
            "conditions_count": len(bot.conditions or []),
            "exit_rules_count": len(bot.exit_rules or []),
        })
    return rows


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
    if payload.desk is not None:
        bot.desk = payload.desk
    if payload.signal_source is not None:
        bot.signal_source = payload.signal_source
    if payload.ml_model_name is not None:
        bot.ml_model_name = payload.ml_model_name
    if payload.ml_confidence_threshold is not None:
        bot.ml_confidence_threshold = payload.ml_confidence_threshold
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
    """Delete a bot."""
    bot = await _get_user_bot(bot_id, current_user.id, db)
    await db.delete(bot)
    await db.commit()
    _maybe_unschedule(bot_id)
    logger.info("Bot deleted", bot_id=bot_id, user_id=current_user.id)


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


# ---------------------------------------------------------------------------
# Analytics — Options Alpha-style bot stats
# ---------------------------------------------------------------------------

@router.get("/{bot_id}/stats", response_model=dict)
async def get_bot_stats(
    bot_id: str,
    days: int = 90,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Options Alpha-style performance stats for a single bot.
    Uses Trade records where strategy_name = bot.name.
    """
    bot = await _get_user_bot(bot_id, current_user.id, db)
    since = datetime.now(UTC) - timedelta(days=days)

    result = await db.execute(
        select(Trade).where(
            Trade.strategy_name == bot.name,
            Trade.closed_at >= since,
        )
    )
    trades = result.scalars().all()

    if not trades:
        return {
            "bot_id": bot_id,
            "bot_name": bot.name,
            "total_trades": 0,
            "win_rate": None,
            "total_pnl": 0.0,
            "avg_pnl": None,
            "avg_win": None,
            "avg_loss": None,
            "profit_factor": None,
            "max_win": None,
            "max_loss": None,
            "avg_hold_hours": None,
            "period_days": days,
        }

    pnls = [float(t.realized_pnl or 0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    hold_seconds = [t.hold_seconds for t in trades if t.hold_seconds]

    return {
        "bot_id": bot_id,
        "bot_name": bot.name,
        "total_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 2),
        "total_pnl": round(sum(pnls), 4),
        "avg_pnl": round(sum(pnls) / len(pnls), 4),
        "avg_win": round(sum(wins) / len(wins), 4) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 4) if losses else None,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 4) if losses and wins else None,
        "max_win": round(max(wins), 4) if wins else None,
        "max_loss": round(min(losses), 4) if losses else None,
        "avg_hold_hours": round(sum(hold_seconds) / len(hold_seconds) / 3600, 2) if hold_seconds else None,
        "period_days": days,
    }


@router.get("/{bot_id}/positions", response_model=list[dict])
async def get_bot_positions(
    bot_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Open paper positions belonging to this bot."""
    bot = await _get_user_bot(bot_id, current_user.id, db)

    result = await db.execute(
        select(Order).where(
            Order.status == "paper",
            Order.raw_payload["bot_id"].as_string() == bot_id,
        )
    )
    orders = result.scalars().all()

    positions = []
    for o in orders:
        raw = o.raw_payload or {}
        entry_price = float(raw.get("entry_price", 0))
        notional = float(o.notional or 1000.0)
        qty = notional / entry_price if entry_price > 0 else 0
        positions.append({
            "order_id": o.id,
            "symbol": o.symbol,
            "side": o.side,
            "entry_price": entry_price,
            "notional": notional,
            "qty": round(qty, 6),
            "take_profit": float(o.take_profit_price) if o.take_profit_price else None,
            "stop_loss": float(o.stop_loss_price) if o.stop_loss_price else None,
            "opened_at": o.created_at.isoformat() if o.created_at else None,
        })
    return positions


@router.get("/{bot_id}/trades", response_model=list[dict])
async def get_bot_trades(
    bot_id: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Trade history for a single bot."""
    bot = await _get_user_bot(bot_id, current_user.id, db)

    result = await db.execute(
        select(Trade)
        .where(Trade.strategy_name == bot.name)
        .order_by(Trade.closed_at.desc())
        .limit(limit)
    )
    trades = result.scalars().all()

    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": float(t.entry_price) if t.entry_price else None,
            "exit_price": float(t.exit_price) if t.exit_price else None,
            "quantity": float(t.quantity) if t.quantity else None,
            "realized_pnl": float(t.realized_pnl) if t.realized_pnl else None,
            "hold_seconds": t.hold_seconds,
            "exit_reason": (t.raw_payload or {}).get("exit_reason"),
            "opened_at": t.opened_at.isoformat() if t.opened_at else None,
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        }
        for t in trades
    ]


