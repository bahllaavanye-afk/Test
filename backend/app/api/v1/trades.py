"""Trade history endpoints."""
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import case, select
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
    limit: int = Query(50, ge=1, le=500),
    symbol: str | None = Query(None, description="Filter by symbol"),
    account_id: str | None = Query(None, description="Filter by account ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Compute avg_fill_price directly in SQL to avoid per‑row Python branching
    avg_fill_price_expr = case(
        (Trade.side == "buy", Trade.entry_price),
        (Trade.side == "sell", Trade.exit_price),
    ).label("avg_fill_price")

    query = (
        select(
            Trade.id,
            Trade.symbol,
            Trade.side,
            Trade.realized_pnl,
            Trade.entry_price,
            Trade.exit_price,
            Trade.quantity,
            Trade.opened_at,
            Trade.closed_at,
            Trade.strategy_name,
            avg_fill_price_expr,
        )
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
    rows = result.all()

    # Use list comprehension for concise and efficient construction
    return [
        TradeOut(
            id=row.id,
            symbol=row.symbol,
            side=row.side,
            realized_pnl=float(row.realized_pnl) if row.realized_pnl is not None else None,
            entry_price=float(row.entry_price) if row.entry_price is not None else None,
            exit_price=float(row.exit_price) if row.exit_price is not None else None,
            avg_fill_price=float(row.avg_fill_price) if row.avg_fill_price is not None else None,
            quantity=float(row.quantity),
            opened_at=row.opened_at,
            closed_at=row.closed_at,
            strategy_name=row.strategy_name,
        )
        for row in rows
    ]