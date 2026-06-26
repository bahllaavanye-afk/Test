"""Portfolio positions endpoint."""
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.account import Account
from app.models.position import Position
from app.models.user import User
from app.utils.logging import logger

# Constants
ALPACA_BROKER = "alpaca"
DEFAULT_QTY = 0
DEFAULT_PRICE = 0
DEFAULT_STRING = ""
DEFAULT_SIDE_LONG = "long"
DEFAULT_SIDE_SHORT = "short"

QUERY_ACCOUNT_ID_DESC = "Filter by account ID"

REDIS_POS_EXIT_PREFIX = "pos_exit:"
REDIS_PRICES_PREFIX = "prices:"
REDIS_TTL_SECONDS = 86400

HTTP_503_SERVICE_UNAVAILABLE = 503
HTTP_404_NOT_FOUND = 404
HTTP_500_INTERNAL_SERVER_ERROR = 500

ERR_REDIS_UNAVAILABLE = "Redis unavailable — exit config cannot be retrieved"
ERR_REDIS_READ_FAILED = "Failed to read exit config from Redis"
ERR_MALFORMED_EXIT_CONFIG = "Malformed exit config in Redis"
ERR_REDIS_UNAVAILABLE_GENERIC = "Redis unavailable"
ERR_REDIS_WRITE_FAILED = "Failed to write exit config to Redis"

DEFAULT_STRATEGY_NAME = "unknown"
DEFAULT_STRATEGY_TYPE = "manual"
DEFAULT_RISK_BUCKET = "directional"

EXIT_STRATEGY_FIXED_STOP_LOSS = "fixed_stop_loss"
EXIT_STRATEGY_FIXED_TAKE_PROFIT = "fixed_take_profit"
EXIT_STRATEGY_TRAILING_STOP = "trailing_stop"
EXIT_STRATEGY_ATR_STOP = "atr_stop"
EXIT_STRATEGY_PROFIT_LOCK = "profit_lock"
EXIT_STRATEGY_REGIME_EXIT = "regime_exit"
EXIT_STRATEGY_ZSCORE_EXIT = "zscore_exit"
EXIT_STRATEGY_TIME_EOD = "time_eod"
EXIT_STRATEGY_MAX_LOSS = "max_loss"

router = APIRouter(prefix="/positions", tags=["positions"])


class PositionOut(BaseModel):
    """Schema representing a position returned by the API."""

    id: str | None = Field(
        default=None,
        description="Unique identifier of the position (asset ID).",
        example="e3c8a0f2-5b6d-4f7e-9c1a-2d5f6b7c8d9e",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol of the asset.",
        example="AAPL",
    )
    quantity: float = Field(
        ...,
        description="Number of shares (positive for long, negative for short).",
        example=150.0,
    )
    avg_cost: float = Field(
        ...,
        description="Average entry price per share.",
        example=145.23,
    )
    current_price: float | None = Field(
        default=None,
        description="Latest market price for the asset.",
        example=148.50,
    )
    unrealized_pnl: float | None = Field(
        default=None,
        description="Unrealized profit and loss in currency units.",
        example=490.5,
    )
    side: str = Field(
        ...,
        description="Position side – either 'long' or 'short'.",
        example="long",
    )

    model_config = ConfigDict(from_attributes=True)

    @validator("side")
    def validate_side(cls, v: str) -> str:
        if v not in {DEFAULT_SIDE_LONG, DEFAULT_SIDE_SHORT}:
            raise ValueError(f"side must be '{DEFAULT_SIDE_LONG}' or '{DEFAULT_SIDE_SHORT}'")
        return v

    @validator("quantity")
    def validate_quantity(cls, v: float) -> float:
        if v == 0:
            raise ValueError("quantity must be non‑zero")
        return v


def _alpaca_position_to_out(p: dict) -> dict:
    """Map an Alpaca REST position dict to PositionOut-compatible shape."""
    qty = float(p.get("qty", DEFAULT_QTY))
    return {
        "id": p.get("asset_id"),
        "symbol": p.get("symbol", DEFAULT_STRING),
        "quantity": qty,
        "avg_cost": float(p.get("avg_entry_price", DEFAULT_PRICE)),
        "current_price": float(p.get("current_price", DEFAULT_PRICE))
        if p.get("current_price")
        else None,
        "unrealized_pnl": float(p.get("unrealized_pl", DEFAULT_PRICE))
        if p.get("unrealized_pl") is not None
        else None,
        "side": DEFAULT_SIDE_LONG if qty >= 0 else DEFAULT_SIDE_SHORT,
    }


@router.get("/", response_model=list[PositionOut])
async def list_positions(
    account_id: str | None = Query(None, description=QUERY_ACCOUNT_ID_DESC),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # If account_id provided, try live Alpaca data for that account
    if account_id:
        acct_result = await db.execute(
            select(Account).where(Account.id == account_id, Account.user_id == current_user.id)
        )
        account = acct_result.scalar_one_or_none()
        if account and account.broker == ALPACA_BROKER and account.encrypted_key:
            from app.brokers.alpaca_orders import get_alpaca_positions

            try:
                live_positions = await get_alpaca_positions(account)
                return [_alpaca_position_to_out(p) for p in live_positions]
            except Exception as e:
                logger.warning(
                    f"Alpaca positions fetch failed: {e} — falling back to DB positions"
                )

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
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=ERR_REDIS_UNAVAILABLE)

    try:
        raw = await redis_client.get(f"{REDIS_POS_EXIT_PREFIX}{symbol}")
    except Exception as exc:
        logger.warning(
            "get_position_exit_config: Redis read failed", symbol=symbol, error=str(exc)
        )
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail=ERR_REDIS_READ_FAILED)

    if not raw:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"No active exit config found for {symbol}",
        )

    try:
        config = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=ERR_MALFORMED_EXIT_CONFIG)

    entry_price = config.get("entry_price")
    peak_price = config.get("peak_price")
    stop_loss = config.get("stop_loss")
    take_profit = config.get("take_profit")
    bars_held = config.get("bars_held", 0)
    strategy_name = config.get("strategy_name", DEFAULT_STRATEGY_NAME)
    strategy_type = config.get("strategy_type", DEFAULT_STRATEGY_TYPE)
    risk_bucket = config.get("risk_bucket", DEFAULT_RISK_BUCKET)
    stored_at = config.get("stored_at")

    # Compute P&L percentage if we have a current price from Redis
    pnl_pct: float | None = None
    try:
        raw_price = await redis_client.get(f"{REDIS_PRICES_PREFIX}{symbol}")
        if raw_price:
            price_data = json.loads(raw_price)
            current_price = float(
                price_data.get("last") or price_data.get("ask") or DEFAULT_PRICE
            )
            if entry_price and current_price:
                pnl_pct = round(
                    (current_price - float(entry_price)) / float(entry_price) * 100, 4
                )
    except Exception:
        pass

    # Determine which exit strategies are active
    exit_strategies_active: list[str] = []
    if stop_loss is not None:
        exit_strategies_active.append(EXIT_STRATEGY_FIXED_STOP_LOSS)
    if take_profit is not None:
        exit_strategies_active.append(EXIT_STRATEGY_FIXED_TAKE_PROFIT)
    if strategy_type == "directional" or risk_bucket == "directional":
        exit_strategies_active.extend(
            [
                EXIT_STRATEGY_TRAILING_STOP,
                EXIT_STRATEGY_ATR_STOP,
                EXIT_STRATEGY_PROFIT_LOCK,
                EXIT_STRATEGY_REGIME_EXIT,
            ]
        )
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


class ExitOptionsUpdate(BaseModel):
    """Schema for updating exit configuration of a position."""

    stop_loss: float | None = Field(
        default=None,
        description="Absolute stop loss price.",
        example=140.0,
        ge=0,
    )
    take_profit: float | None = Field(
        default=None,
        description="Absolute take profit price.",
        example=160.0,
        ge=0,
    )
    profit_target_pct: float | None = Field(
        default=None,
        description="Target profit expressed as a percentage of entry price.",
        example=10.0,
        ge=0,
        le=100,
    )
    stop_loss_pct: float | None = Field(
        default=None,
        description="Stop loss expressed as a percentage of entry price.",
        example=5.0,
        ge=0,
        le=100,
    )
    trailing_stop_pct: float | None = Field(
        default=None,
        description="Trailing stop distance as a percentage.",
        example=2.5,
        ge=0,
        le=100,
    )
    expiration_days: int | None = Field(
        default=None,
        description="Number of days after which the position should be closed.",
        example=30,
        ge=1,
    )
    pricing_method: str | None = Field(
        default=None,
        description="Pricing method to use for order execution (e.g., 'mid', 'bid', 'ask').",
        example="mid",
    )
    bid_ask: str | None = Field(
        default=None,
        description="Preferred side for execution when using bid/ask pricing.",
        example="bid",
    )

    @validator("pricing_method")
    def validate_pricing_method(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"mid", "bid", "ask"}
        if v not in allowed:
            raise ValueError(f"pricing_method must be one of {allowed}")
        return v

    @validator("bid_ask")
    def validate_bid_ask(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"bid", "ask"}
        if v not in allowed:
            raise ValueError(f"bid_ask must be one of {allowed}")
        return v

    model_config = ConfigDict(extra="forbid")