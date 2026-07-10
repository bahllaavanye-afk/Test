"""Trade history endpoints."""
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.account import Account
from app.models.trade import Trade
from app.models.user import User

router = APIRouter(prefix="/trades", tags=["trades"])


class TradeOut(BaseModel):
    """Schema representing a trade record returned by the API."""

    id: str = Field(..., description="Unique identifier for the trade.", json_schema_extra={"example": "trd_12345"})
    symbol: str = Field(..., description="Ticker symbol of the traded instrument.", json_schema_extra={"example": "AAPL"})
    side: str = Field(
        ...,
        description="Trade direction; either 'buy' or 'sell'.",
        json_schema_extra={"example": "buy"},
    )
    realized_pnl: float | None = Field(
        None,
        description="Realized profit and loss in the account's base currency.",
        json_schema_extra={"example": 152.35},
    )
    entry_price: float | None = Field(
        None,
        description="Price at which the position was entered.",
        json_schema_extra={"example": 145.30},
    )
    exit_price: float | None = Field(
        None,
        description="Price at which the position was exited.",
        json_schema_extra={"example": 150.00},
    )
    avg_fill_price: float | None = Field(
        None,
        description=(
            "Average fill price used for chart markers. "
            "When a dedicated fill-price column is unavailable, "
            "the entry price is used for buys and the exit price for sells."
        ),
        json_schema_extra={"example": 145.30},
    )
    quantity: float = Field(
        ...,
        description="Number of shares/contracts traded.",
        json_schema_extra={"example": 100},
    )
    opened_at: datetime | None = Field(
        None,
        description="Timestamp when the trade was opened.",
        json_schema_extra={"example": "2023-01-01T09:30:00Z"},
    )
    closed_at: datetime | None = Field(
        None,
        description="Timestamp when the trade was closed.",
        json_schema_extra={"example": "2023-01-01T15:45:00Z"},
    )
    strategy_name: str | None = Field(
        None,
        description="Name of the strategy that generated the trade.",
        json_schema_extra={"example": "mean_rev_20_2"},
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Ensure side is either 'buy' or 'sell'."""
        if v not in {"buy", "sell"}:
            raise ValueError("side must be either 'buy' or 'sell'")
        return v

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: float) -> float:
        """Quantity must be a positive number."""
        if v <= 0:
            raise ValueError("quantity must be greater than 0")
        return v


@router.get("/", response_model=list[TradeOut])
async def list_trades(
    limit: int | None = Query(50, ge=1, le=500),
    symbol: str | None = Query(None, description="Filter by symbol"),
    account_id: str | None = Query(None, description="Filter by account ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve a list of trades for the current user.

    Edge‑case handling:
    * `limit` may be `None` or out of bounds – we coerce it to a safe default.
    * Empty result sets are returned as an empty list.
    * Trades with missing critical fields (`side` or `quantity`) are skipped.
    """
    # Guard against unexpected None or out‑of‑range values.
    if limit is None or limit < 1:
        limit = 1
    if limit > 500:
        limit = 500

    query = (
        select(Trade)
        .join(Account, Trade.account_id == Account.id)
        .where(Account.user_id == current_user.id)
        .order_by(Trade.opened_at.desc())
        .limit(limit)
    )
    if account_id:
        query = query.where(Trade.account_id == account_id)
    if symbol:
        query = query.where(Trade.symbol == symbol)

    result = await db.execute(query)
    trades = result.scalars().all()

    # Return early for an empty collection.
    if not trades:
        return []

    out: list[TradeOut] = []
    for t in trades:
        # Skip records that lack essential data required by the schema.
        if t.side not in {"buy", "sell"} or t.quantity is None:
            continue

        # Determine fill price safely.
        if t.side == "buy":
            fill_price = float(t.entry_price) if t.entry_price is not None else None
        else:
            fill_price = float(t.exit_price) if t.exit_price is not None else None

        out.append(
            TradeOut(
                id=t.id,
                symbol=t.symbol,
                side=t.side,
                realized_pnl=float(t.realized_pnl) if t.realized_pnl is not None else None,
                entry_price=float(t.entry_price) if t.entry_price is not None else None,
                exit_price=float(t.exit_price) if t.exit_price is not None else None,
                avg_fill_price=fill_price,
                quantity=float(t.quantity),
                opened_at=t.opened_at,
                closed_at=t.closed_at,
                strategy_name=t.strategy_name,
            )
        )
    return out