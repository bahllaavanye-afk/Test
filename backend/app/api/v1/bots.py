"""Bot Builder API — CRUD + manual run + toggle."""
from __future__ import annotations

from datetime import UTC, datetime

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


@router.get("/{bot_id}/performance", response_model=dict)
async def get_bot_performance(
    bot_id: str,
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Option Alpha-style per-bot performance: cumulative realized P&L series + stats.

    Built from real closed Trade rows attributed to this bot (strategy_name ==
    bot.name, written by check_bot_exits). Honest: an empty series means the bot
    hasn't closed a position yet — nothing is fabricated. Powers the per-bot
    sparkline in BotBuilder (the requested screenshot-parity graph).
    """
    from datetime import datetime, timedelta, timezone

    from app.models.trade import Trade

    bot = await _get_user_bot(bot_id, current_user.id, db)
    days = max(1, min(days, 365))
    since = datetime.now(timezone.utc) - timedelta(days=days)

    rows = (await db.execute(
        select(Trade)
        .where(Trade.strategy_name == bot.name, Trade.closed_at >= since)
        .order_by(Trade.closed_at.asc())
    )).scalars().all()

    series: list[dict] = []
    cum = 0.0
    wins = 0
    peak = 0.0
    max_dd = 0.0
    high_pnl = 0.0
    low_pnl = 0.0
    gross_win = 0.0
    gross_loss = 0.0
    holds: list[int] = []
    # Day P/L = cumulative from trades closed today only (OA "Day P/L").
    today = datetime.now(timezone.utc).date()
    day_pnl = 0.0
    for t in rows:
        pnl = float(t.realized_pnl or 0)
        cum += pnl
        if pnl > 0:
            wins += 1
            gross_win += pnl
        else:
            gross_loss += -pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        high_pnl = max(high_pnl, cum)
        low_pnl = min(low_pnl, cum)
        if t.closed_at and t.closed_at.date() == today:
            day_pnl += pnl
        if t.hold_seconds:
            holds.append(int(t.hold_seconds))
        series.append({
            "date": t.closed_at.isoformat() if t.closed_at else None,
            "pnl": round(pnl, 2),
            "cum_pnl": round(cum, 2),
            "symbol": t.symbol,
        })

    n = len(rows)
    losses = n - wins
    alloc = float(getattr(bot, "allocation", 0) or 0) or None

    # Sharpe / Sortino on per-trade returns (OA "Analyze" sidebar). Unitless,
    # annualization-free ratios over the closed-trade P&L sequence — enough to
    # rank consistency; None when too few trades to be meaningful.
    import statistics
    pnls = [float(t.realized_pnl or 0) for t in rows]
    sharpe = sortino = None
    if len(pnls) >= 2:
        mean = statistics.fmean(pnls)
        sd = statistics.pstdev(pnls)
        downside = statistics.pstdev([min(0.0, p) for p in pnls])
        sharpe = round(mean / sd, 2) if sd else None
        sortino = round(mean / downside, 2) if downside else None

    # OA "Analyze" breakdowns: P&L grouped by weekday, entry hour, and symbol.
    from collections import defaultdict
    by_weekday: dict[int, float] = defaultdict(float)
    by_hour: dict[int, float] = defaultdict(float)
    by_symbol: dict[str, float] = defaultdict(float)
    for t in rows:
        pnl = float(t.realized_pnl or 0)
        when = t.opened_at or t.closed_at
        if when:
            by_weekday[when.weekday()] += pnl
            by_hour[when.hour] += pnl
        by_symbol[t.symbol] += pnl
    _wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    breakdown = {
        "by_weekday": [{"label": _wd[k], "pnl": round(v, 2)} for k, v in sorted(by_weekday.items())],
        "by_hour": [{"label": f"{h:02d}:00", "pnl": round(v, 2)} for h, v in sorted(by_hour.items())],
        "by_symbol": [{"label": k, "pnl": round(v, 2)}
                      for k, v in sorted(by_symbol.items(), key=lambda kv: -kv[1])[:12]],
    }

    # Current win/loss streak (OA "Streak"): sign + length of the trailing run.
    streak_n = 0
    streak_kind: str | None = None
    for t in reversed(rows):
        won = float(t.realized_pnl or 0) > 0
        kind = "wins" if won else "losses"
        if streak_kind is None:
            streak_kind, streak_n = kind, 1
        elif kind == streak_kind:
            streak_n += 1
        else:
            break

    # Capital block (OA sidebar). Net liquid = allocation + realized P/L. At-risk /
    # maintenance require live open-option margin which the paper engine doesn't
    # track per-bot yet, so they're reported as 0 rather than guessed.
    net_liquid = round((alloc or 0) + cum, 2) if alloc is not None else None
    change_pnl = day_pnl  # OA "Change" = day P/L vs prior close
    # OA metrics: Profit Factor = gross win / gross loss; Return % = P/L / allocation.
    return {
        "bot_id": bot.id,
        "bot_name": bot.name,
        "days": days,
        "series": series,
        "total_pnl": round(cum, 2),
        "total_pnl_pct": round(cum / alloc * 100, 2) if alloc else None,
        "day_pnl": round(day_pnl, 2),
        "change_pnl": round(change_pnl, 2),
        "change_pct": round(change_pnl / net_liquid * 100, 2) if net_liquid else None,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / n, 4) if n else None,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else (None if not gross_win else 99.99),
        "avg_win": round(gross_win / wins, 2) if wins else None,
        "avg_loss": round(-gross_loss / losses, 2) if losses else None,
        "avg_pnl": round(cum / n, 2) if n else None,
        "high_pnl": round(high_pnl, 2),
        "low_pnl": round(low_pnl, 2),
        "max_drawdown": round(max_dd, 2),
        "streak": streak_n,
        "streak_kind": streak_kind,
        "sharpe": sharpe,
        "sortino": sortino,
        "avg_hold_hours": round(sum(holds) / len(holds) / 3600, 2) if holds else None,
        "allocation": alloc,
        "net_liquid": net_liquid,
        "at_risk": 0.0,
        "available": net_liquid,
        "maintenance": 0.0,
        "breakdown": breakdown,
    }


@router.get("/{bot_id}/activity", response_model=dict)
async def get_bot_activity(
    bot_id: str,
    limit: int = 25,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Option Alpha-style per-bot detail: open positions + recent trade history.

    Open positions = this bot's still-open paper orders (raw_payload.bot_id,
    written by the bot engine). Trade history = closed Trade rows attributed by
    strategy_name == bot.name (same attribution the performance graph uses).
    Honest: empty lists mean the bot hasn't traded — nothing is fabricated.
    """
    from app.models.order import Order
    from app.models.trade import Trade

    bot = await _get_user_bot(bot_id, current_user.id, db)
    limit = max(1, min(limit, 100))

    open_rows = (await db.execute(
        select(Order)
        .where(Order.status == "paper")
        .order_by(Order.created_at.desc())
        .limit(500)
    )).scalars().all()
    open_positions = [
        {
            "order_id": o.id,
            "symbol": o.symbol,
            "side": o.side,
            "entry_price": float((o.raw_payload or {}).get("entry_price", 0) or 0),
            "notional": float((o.raw_payload or {}).get("notional", 0) or 0),
            "take_profit": float(o.take_profit_price) if o.take_profit_price is not None else None,
            "stop_loss": float(o.stop_loss_price) if o.stop_loss_price is not None else None,
            "opened_at": o.created_at.isoformat() if o.created_at else None,
        }
        for o in open_rows
        if (o.raw_payload or {}).get("bot_id") == bot.id
    ]

    trades = (await db.execute(
        select(Trade)
        .where(Trade.strategy_name == bot.name)
        .order_by(Trade.closed_at.desc())
        .limit(limit)
    )).scalars().all()
    trade_history = [
        {
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": float(t.entry_price),
            "exit_price": float(t.exit_price),
            "quantity": float(t.quantity),
            "realized_pnl": round(float(t.realized_pnl or 0), 2),
            "exit_reason": (t.raw_payload or {}).get("exit_reason"),
            "opened_at": t.opened_at.isoformat() if t.opened_at else None,
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        }
        for t in trades
    ]

    return {
        "bot_id": bot.id,
        "bot_name": bot.name,
        "open_positions": open_positions,
        "trade_history": trade_history,
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
