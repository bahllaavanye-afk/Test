"""
Copy Trading Desk — mirror signals from top-performing strategies.

Leaderboard ranks internal strategies by 30-day rolling Sharpe.
Following a strategy auto-mirrors its signals at configurable size multiplier.
"""
import json
from datetime import UTC, datetime, timedelta
from typing import Dict, List

import numpy as np
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models.strategy import Strategy
from app.models.trade import Trade
from app.models.user import User

router = APIRouter(prefix="/copy-trading", tags=["copy-trading"])


class FollowRequest(BaseModel):
    """Request payload for following a strategy."""

    strategy_id: str = Field(
        ...,
        description="Unique identifier of the strategy to follow.",
        example="strategy_abc123",
    )
    size_multiplier: float = Field(
        1.0,
        gt=0,
        le=10,
        description="Position size multiplier applied to the original signal (0–10×).",
        example=1.5,
    )

    @validator("strategy_id")
    def _non_empty_strategy_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("strategy_id must be a non‑empty string")
        return v


class FollowResponse(BaseModel):
    """Response model for a successful follow operation."""

    followed: bool = Field(
        True,
        description="Indicates whether the follow request succeeded.",
        example=True,
    )
    strategy_id: str = Field(
        ...,
        description="ID of the strategy that was followed.",
        example="strategy_abc123",
    )
    size_multiplier: float = Field(
        ...,
        description="The multiplier applied to the strategy’s position size.",
        example=1.5,
    )


class UnfollowResponse(BaseModel):
    """Response model for a successful unfollow operation."""

    unfollowed: bool = Field(
        True,
        description="Indicates whether the unfollow request succeeded.",
        example=True,
    )
    strategy_id: str = Field(
        ...,
        description="ID of the strategy that was unfollowed.",
        example="strategy_abc123",
    )


class ListFollowsResponse(BaseModel):
    """Response model for listing all followed strategies."""

    follows: Dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of strategy IDs to their size multipliers.",
        example={"strategy_abc123": 1.2, "strategy_xyz789": 0.8},
    )


class LeaderboardEntry(BaseModel):
    """Single entry of the copy‑trading leaderboard."""

    strategy_id: int = Field(..., description="Database identifier of the strategy.", example=42)
    name: str = Field(..., description="Internal name of the strategy.", example="mean_rev_20_1.5")
    display_name: str = Field(..., description="Human‑readable name.", example="Mean Rev 20 1.5")
    market_type: str = Field(..., description="Market classification (e.g., equities, crypto).", example="equities")
    strategy_type: str = Field(..., description="Category of the strategy.", example="mean_reversion")
    risk_bucket: str = Field(..., description="Risk classification bucket.", example="medium")
    is_enabled: bool = Field(..., description="Whether the strategy is currently active.", example=True)
    symbols: List[str] = Field(..., description="List of symbols the strategy trades.", example=["AAPL", "MSFT"])
    total_pnl: float = Field(..., description="Total realized P&L over the window (USD).", example=12345.67)
    return_pct: float = Field(..., description="Return as a percentage of allocated capital.", example=4.56)
    win_rate: float = Field(..., description="Winning trade percentage.", example=62.5)
    total_trades: int = Field(..., description="Number of trades executed in the window.", example=210)
    sharpe_30d: float = Field(..., description="30‑day rolling Sharpe ratio.", example=1.23)
    allocation: float = Field(..., description="Allocated capital for the strategy (USD).", example=2500.0)
    rank: int = Field(..., description="Leaderboard rank (1 = best).", example=1)


async def _compute_leaderboard(db: AsyncSession, days: int = 30) -> List[Dict]:
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
    daily_pnl: dict[int, List[float]] = {}
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

    rows: List[Dict] = []
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

        rows.append(
            {
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
            }
        )

    # Sort by Sharpe descending
    rows.sort(key=lambda x: x["sharpe_30d"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


@router.get("/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    days: int = Query(30, ge=7, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Rank all strategies by rolling Sharpe for the last N days."""
    return await _compute_leaderboard(db, days)


def _follows_key(user_id: str) -> str:
    return f"copy_trading:follows:{user_id}"


@router.post("/follow", response_model=FollowResponse)
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

    return FollowResponse(
        followed=True,
        strategy_id=body.strategy_id,
        size_multiplier=body.size_multiplier,
    )


@router.delete("/follow/{strategy_id}", response_model=UnfollowResponse)
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

    return UnfollowResponse(unfollowed=True, strategy_id=strategy_id)


@router.get("/follows", response_model=ListFollowsResponse)
async def list_follows(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all strategies the current user is following, with their size multipliers."""
    from app.redis_client import get_redis

    redis_client = get_redis()
    if redis_client is None:
        return ListFollowsResponse(follows={})

    try:
        key = _follows_key(str(current_user.id))
        raw = await redis_client.get(key)
        follows: dict = json.loads(raw) if raw else {}
    except Exception:
        return ListFollowsResponse(follows={})

    return ListFollowsResponse(follows=follows)