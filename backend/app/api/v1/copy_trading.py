"""
Copy Trading Desk — mirror signals from top-performing strategies.

Leaderboard ranks internal strategies by 30-day rolling Sharpe.
Following a strategy auto-mirrors its signals at configurable size multiplier.
"""
import json
from datetime import UTC, datetime, timedelta

import numpy as np
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models.strategy import Strategy
from app.models.trade import Trade
from app.models.user import User

router = APIRouter(prefix="/copy-trading", tags=["copy-trading"])


async def _compute_leaderboard(db: AsyncSession, days: int = 30) -> list[dict]:
    """Rank strategies by rolling Sharpe, win rate, and total P&L over last N days."""
    cutoff = datetime.now(UTC) - timedelta(days=days)

    strategies_result = await db.execute(select(Strategy))
    strategies = strategies_result.scalars().all()

    # Per-strategy trade stats over the window
    stats_result = await db.execute(
        select(
            Trade.strategy_id,
            func.count(Trade.id).label("total_trades"),
            func.sum(Trade.realized_pnl).label("total_pnl"),
            func.avg(Trade.realized_pnl).label("avg_pnl"),
        )
        .where(Trade.closed_at >= cutoff)
        .where(Trade.realized_pnl.isnot(None))
        .group_by(Trade.strategy_id)
    )
    stats_map = {r.strategy_id: r for r in stats_result}

    # Win count
    wins_result = await db.execute(
        select(Trade.strategy_id, func.count(Trade.id).label("wins"))
        .where(Trade.closed_at >= cutoff)
        .where(Trade.realized_pnl > 0)
        .group_by(Trade.strategy_id)
    )
    wins_map = {r.strategy_id: r.wins for r in wins_result}

    # Daily P&L series for Sharpe calculation
    # date_trunc is PostgreSQL-specific; fall back gracefully for SQLite (dev/test)
    daily_pnl: dict[str, list[float]] = {}
    try:
        daily_result = await db.execute(
            select(
                Trade.strategy_id,
                func.date_trunc("day", Trade.closed_at).label("day"),
                func.sum(Trade.realized_pnl).label("day_pnl"),
            )
            .where(Trade.closed_at >= cutoff)
            .where(Trade.realized_pnl.isnot(None))
            .group_by(Trade.strategy_id, func.date_trunc("day", Trade.closed_at))
            .order_by(Trade.strategy_id, func.date_trunc("day", Trade.closed_at))
        )
        for r in daily_result:
            daily_pnl.setdefault(r.strategy_id, []).append(float(r.day_pnl or 0))
    except Exception:
        # SQLite fallback: group by date string cast
        try:
            daily_result = await db.execute(
                select(
                    Trade.strategy_id,
                    func.strftime("%Y-%m-%d", Trade.closed_at).label("day"),
                    func.sum(Trade.realized_pnl).label("day_pnl"),
                )
                .where(Trade.closed_at >= cutoff)
                .where(Trade.realized_pnl.isnot(None))
                .group_by(Trade.strategy_id, func.strftime("%Y-%m-%d", Trade.closed_at))
                .order_by(Trade.strategy_id, func.strftime("%Y-%m-%d", Trade.closed_at))
            )
            for r in daily_result:
                daily_pnl.setdefault(r.strategy_id, []).append(float(r.day_pnl or 0))
        except Exception:
            # If both fail, Sharpe will be 0.0 for all strategies
            pass

    rows = []
    for s in strategies:
        st = stats_map.get(s.id)
        if not st or not st.total_trades:
            continue

        total_pnl = float(st.total_pnl or 0)
        total_trades = int(st.total_trades)
        wins = wins_map.get(s.id, 0)
        win_rate = round(wins / total_trades * 100, 1) if total_trades > 0 else 0.0

        allocation = float(s.params.get("allocation_usd", 2500.0)) if s.params else 2500.0
        returns = daily_pnl.get(s.id, [])

        if len(returns) >= 5:
            arr = np.array(returns, dtype=float)
            ret_pct = arr / max(allocation, 1)
            sharpe = float(np.mean(ret_pct) / (np.std(ret_pct, ddof=1) + 1e-9) * np.sqrt(252))
        else:
            sharpe = 0.0

        rows.append({
            "strategy_id": s.id,
            "name": s.name,
            "display_name": s.display_name or s.name.replace("_", " ").title(),
            "market_type": s.market_type,
            "strategy_type": s.strategy_type,
            "risk_bucket": s.risk_bucket,
            "is_enabled": s.is_enabled,
            "symbols": s.symbols,
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(total_pnl / allocation * 100, 2) if allocation > 0 else 0.0,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "sharpe_30d": round(sharpe, 3),
            "allocation": allocation,
        })

    # Sort by Sharpe descending
    rows.sort(key=lambda x: x["sharpe_30d"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


class FollowRequest(BaseModel):
    strategy_id: str = Field(..., description="ID of the strategy to follow")
    size_multiplier: float = Field(1.0, gt=0, le=10, description="Position size multiplier (0–10×)")


@router.get("/leaderboard")
async def get_leaderboard(
    days: int = Query(30, ge=7, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Rank all strategies by rolling Sharpe for the last N days."""
    return await _compute_leaderboard(db, days)


def _follows_key(user_id: str) -> str:
    return f"copy_trading:follows:{user_id}"


@router.post("/follow")
async def follow_strategy(
    body: FollowRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Follow a strategy and mirror its signals at the given size multiplier."""
    from app.redis_client import get_redis

    redis_client = get_redis()
    if redis_client is not None:
        try:
            key = _follows_key(str(current_user.id))
            raw = await redis_client.get(key)
            follows: dict = json.loads(raw) if raw else {}
            follows[body.strategy_id] = body.size_multiplier
            await redis_client.set(key, json.dumps(follows))
        except Exception:
            # Optimistic — Redis write failure is non-fatal
            pass

    return {
        "followed": True,
        "strategy_id": body.strategy_id,
        "size_multiplier": body.size_multiplier,
    }


@router.delete("/follow/{strategy_id}")
async def unfollow_strategy(
    strategy_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Stop mirroring a strategy's signals."""
    from app.redis_client import get_redis

    redis_client = get_redis()
    if redis_client is not None:
        try:
            key = _follows_key(str(current_user.id))
            raw = await redis_client.get(key)
            follows: dict = json.loads(raw) if raw else {}
            follows.pop(strategy_id, None)
            await redis_client.set(key, json.dumps(follows))
        except Exception:
            # Optimistic — Redis write failure is non-fatal
            pass

    return {"unfollowed": True, "strategy_id": strategy_id}


@router.get("/follows")
async def list_follows(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all strategies the current user is following, with their size multipliers."""
    from app.redis_client import get_redis

    redis_client = get_redis()
    if redis_client is None:
        return {"follows": []}

    try:
        key = _follows_key(str(current_user.id))
        raw = await redis_client.get(key)
        follows: dict = json.loads(raw) if raw else {}
    except Exception:
        return {"follows": []}

    return {
        "follows": [
            {"strategy_id": sid, "size_multiplier": mult}
            for sid, mult in follows.items()
        ]
    }
