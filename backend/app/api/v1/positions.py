"""Portfolio positions endpoint."""
import json
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.position import Position
from app.models.user import User
from app.models.account import Account
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from app.utils.logging import logger

# Constants
ROUTER_PREFIX = "/positions"
ROUTER_TAGS = ["positions"]
BROKER_ALPACA = "alpaca"
DEFAULT_ZERO = 0
DEFAULT_STRATEGY_NAME = "unknown"
DEFAULT_STRATEGY_TYPE = "manual"
DEFAULT_RISK_BUCKET = "directional"

REDIS_POS_EXIT_KEY = "pos_exit:{symbol}"
REDIS_PRICES_KEY = "prices:{symbol}"

HTTP_503_SERVICE_UNAVAILABLE = 503
HTTP_404_NOT_FOUND = 404
HTTP_500_INTERNAL_ERROR = 500

EXIT_STRATEGY_FIXED_STOP_LOSS = "fixed_stop_loss"
EXIT_STRATEGY_FIXED_TAKE_PROFIT = "fixed_take_profit"
EXIT_STRATEGY_TRAILING_STOP = "trailing_stop"
EXIT_STRATEGY_ATR_STOP = "atr_stop"
EXIT_STRATEGY_PROFIT_LOCK = "profit_lock"
EXIT_STRATEGY_REGIME_EXIT = "regime_exit"
EXIT_STRATEGY_ZSCORE_EXIT = "zscore_exit"
EXIT_STRATEGY_TIME_EOD = "time_eod"
EXIT_STRATEGY_MAX_LOSS = "max_loss"

router = APIRouter(prefix=ROUTER_PREFIX, tags=ROUTER_TAGS)


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
    qty = float(p.get("qty", DEFAULT_ZERO))
    return {
        "id": p.get("asset_id"),
        "symbol": p.get("symbol", ""),
        "quantity": qty,
        "avg_cost": float(p.get("avg_entry_price", DEFAULT_ZERO)),
        "current_price": float(p.get("current_price", DEFAULT_ZERO)) if p.get("current_price") else None,
        "unrealized_pnl": float(p.get("unrealized_pl", DEFAULT_ZERO)) if p.get("unrealized_pl") is not None else None,
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
        if account and account.broker == BROKER_ALPACA and account.encrypted_key:
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
        .where(Position.quantity != DEFAULT_ZERO)
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
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis unavailable — exit config cannot be retrieved",
        )

    try:
        raw = await redis_client.get(REDIS_POS_EXIT_KEY.format(symbol=symbol))
    except Exception as exc:
        logger.warning("get_position_exit_config: Redis read failed", symbol=symbol, error=str(exc))
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="Failed to read exit config from Redis")

    if not raw:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"No active exit config found for {symbol}",
        )

    try:
        config = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=HTTP_500_INTERNAL_ERROR, detail="Malformed exit config in Redis")

    entry_price = config.get("entry_price")
    peak_price = config.get("peak_price")
    stop_loss = config.get("stop_loss")
    take_profit = config.get("take_profit")
    bars_held = config.get("bars_held", DEFAULT_ZERO)
    strategy_name = config.get("strategy_name", DEFAULT_STRATEGY_NAME)
    strategy_type = config.get("strategy_type", DEFAULT_STRATEGY_TYPE)
    risk_bucket = config.get("risk_bucket", DEFAULT_RISK_BUCKET)
    stored_at = config.get("stored_at")

    # Compute P&L percentage if we have a current price from Redis
    pnl_pct: float | None = None
    try:
        raw_price = await redis_client.get(REDIS_PRICES_KEY.format(symbol=symbol))
        if raw_price:
            price_data = json.loads(raw_price)
            current_price = float(price_data.get("last") or price_data.get("ask") or DEFAULT_ZERO)
            if entry_price and current_price:
                pnl_pct = round((current_price - float(entry_price)) / float(entry_price) * 100, 4)
    except Exception:
        pass

    # Determine which exit strategies are active
    exit_strategies_active: list[str] = []
    if stop_loss is not None:
        exit_strategies_active.append(EXIT_STRATEGY_FIXED_STOP_LOSS)
    if take_profit is not None:
        exit_strategies_active.append(EXIT_STRATEGY_FIXED_TAKE_PROFIT)
    if strategy_type == "directional" or risk_bucket == "directional":
        exit_strategies_active.extend([
            EXIT_STRATEGY_TRAILING_STOP,
            EXIT_STRATEGY_ATR_STOP,
            EXIT_STRATEGY_PROFIT_LOCK,
            EXIT_STRATEGY_REGIME_EXIT,
        ])
    if strategy_type == "arbitrage" or risk_bucket == "arbitrage":
        exit_strategies_active.extend([EXIT_STRATEGY_ZSCORE_EXIT, EXIT_STRATEGY_TIME_EOD])
    exit_strategies_active.append(EXIT_STRATEGY_MAX_LOSS)

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