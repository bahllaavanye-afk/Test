"""Portfolio positions endpoint."""
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.account import Account
from app.models.position import Position
from app.models.user import User
from app.utils.logging import logger

router = APIRouter(prefix="/positions", tags=["positions"])


class PositionOut(BaseModel):
    id: str | None = None
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float | None
    unrealized_pnl: float | None
    side: str

    model_config = ConfigDict(from_attributes=True)


def _alpaca_position_to_out(p: dict) -> dict:
    """Map an Alpaca REST position dict to PositionOut-compatible shape."""
    qty = float(p.get("qty", 0))
    return {
        "id": p.get("asset_id"),
        "symbol": p.get("symbol", ""),
        "quantity": qty,
        "avg_cost": float(p.get("avg_entry_price", 0)),
        "current_price": float(p.get("current_price", 0)) if p.get("current_price") else None,
        "unrealized_pnl": float(p.get("unrealized_pl", 0)) if p.get("unrealized_pl") is not None else None,
        "side": "long" if qty >= 0 else "short",
    }


@router.get("/", response_model=list[PositionOut])
async def list_positions(
    account_id: str | None = Query(None, description="Filter by account ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # If account_id provided, try live Alpaca data for that account
    if account_id:
        acct_result = await db.execute(
            select(Account).where(Account.id == account_id, Account.user_id == current_user.id)
        )
        account = acct_result.scalar_one_or_none()
        if account and account.broker == "alpaca" and account.encrypted_key:
            from app.brokers.alpaca_orders import get_alpaca_positions
            try:
                live_positions = await get_alpaca_positions(account)
                return [_alpaca_position_to_out(p) for p in live_positions]
            except Exception as e:
                logger.warning(f"Alpaca positions fetch failed: {e} — falling back to DB positions")

    # Fall back to DB positions
    query = (
        select(Position)
        .join(Account, Position.account_id == Account.id)
        .where(Account.user_id == current_user.id)
        .where(Position.quantity != 0)
    )
    if account_id:
        query = query.where(Position.account_id == account_id)

    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{symbol}/exit-config")
async def get_position_exit_config(
    symbol: str,
    current_user: User = Depends(get_current_user),
):
    """Return the active exit conditions for an open position.

    Reads the exit config stored in Redis under key pos_exit:{symbol}.
    Returns the entry price, stop loss, take profit, peak price, bars held,
    and current P&L percentage.
    """
    from app.redis_client import get_redis

    redis_client = get_redis()
    if redis_client is None:
        raise HTTPException(
            status_code=503,
            detail="Redis unavailable — exit config cannot be retrieved",
        )

    try:
        raw = await redis_client.get(f"pos_exit:{symbol}")
    except Exception as exc:
        logger.warning("get_position_exit_config: Redis read failed", symbol=symbol, error=str(exc))
        raise HTTPException(status_code=503, detail="Failed to read exit config from Redis")

    if not raw:
        raise HTTPException(
            status_code=404,
            detail=f"No active exit config found for {symbol}",
        )

    try:
        config = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="Malformed exit config in Redis")

    entry_price = config.get("entry_price")
    peak_price = config.get("peak_price")
    stop_loss = config.get("stop_loss")
    take_profit = config.get("take_profit")
    bars_held = config.get("bars_held", 0)
    strategy_name = config.get("strategy_name", "unknown")
    strategy_type = config.get("strategy_type", "manual")
    risk_bucket = config.get("risk_bucket", "directional")
    stored_at = config.get("stored_at")

    # Compute P&L percentage if we have a current price from Redis
    pnl_pct: float | None = None
    try:
        raw_price = await redis_client.get(f"prices:{symbol}")
        if raw_price:
            price_data = json.loads(raw_price)
            current_price = float(price_data.get("last") or price_data.get("ask") or 0)
            if entry_price and current_price:
                pnl_pct = round((current_price - float(entry_price)) / float(entry_price) * 100, 4)
    except Exception:
        pass

    # Determine which exit strategies are active
    exit_strategies_active: list[str] = []
    if stop_loss is not None:
        exit_strategies_active.append("fixed_stop_loss")
    if take_profit is not None:
        exit_strategies_active.append("fixed_take_profit")
    if strategy_type == "directional" or risk_bucket == "directional":
        exit_strategies_active.extend(["trailing_stop", "atr_stop", "profit_lock", "regime_exit"])
    if strategy_type == "arbitrage" or risk_bucket == "arbitrage":
        exit_strategies_active.extend(["zscore_exit", "time_eod"])
    exit_strategies_active.append("max_loss")

    return {
        "symbol": symbol,
        "strategy_name": strategy_name,
        "strategy_type": strategy_type,
        "risk_bucket": risk_bucket,
        "exit_strategies_active": exit_strategies_active,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "peak_price": peak_price,
        "bars_held": bars_held,
        "pnl_pct": pnl_pct,
        "stored_at": stored_at,
    }


class ExitOptionsUpdate(BaseModel):
    stop_loss: float | None = None
    take_profit: float | None = None
    profit_target_pct: float | None = None
    stop_loss_pct: float | None = None
    trailing_stop_pct: float | None = None
    expiration_days: int | None = None
    pricing_method: str | None = None
    bid_ask_guard: bool | None = None
    notes: str | None = None
    tags: list[str] | None = None


@router.patch("/{symbol}/exit-config")
async def update_position_exit_config(
    symbol: str,
    body: ExitOptionsUpdate,
    current_user: User = Depends(get_current_user),
):
    """Update the active exit conditions for an open position in Redis."""
    from app.redis_client import get_redis

    redis_client = get_redis()
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    try:
        raw = await redis_client.get(f"pos_exit:{symbol}")
    except Exception:
        raise HTTPException(status_code=503, detail="Failed to read exit config from Redis")

    config: dict = json.loads(raw) if raw else {}

    updates = body.model_dump(exclude_none=True)
    config.update(updates)

    try:
        await redis_client.set(f"pos_exit:{symbol}", json.dumps(config), ex=86400)
    except Exception:
        raise HTTPException(status_code=503, detail="Failed to write exit config to Redis")

    return {"symbol": symbol, "updated": True, "config": config}
