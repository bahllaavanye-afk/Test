"""Trade history endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.account import Account
from app.models.trade import Trade
from app.models.user import User
from pydantic import BaseModel, ConfigDict
from datetime import datetime

router = APIRouter(prefix="/trades", tags=["trades"])


class TradeOut(BaseModel):
    id: str
    symbol: str
    side: str
    realized_pnl: float | None
    entry_price: float | None
    exit_price: float | None
    # avg_fill_price: best-effort fill price for chart markers.
    # Uses entry_price for buys and exit_price for sells as a proxy when
    # a dedicated fill-price column is not yet present on the model.
    avg_fill_price: float | None
    quantity: float
    opened_at: datetime | None
    closed_at: datetime | None
    strategy_name: str | None

    model_config = ConfigDict(from_attributes=True)


@router.get("/", response_model=list[TradeOut])
async def list_trades(
    limit: int = Query(50, ge=1, le=500),
    symbol: str | None = Query(None, description="Filter by symbol"),
    account_id: str | None = Query(None, description="Filter by account ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
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

    # Build response manually so we can compute avg_fill_price from existing columns
    out: list[TradeOut] = []
    for t in trades:
        fill_price: float | None
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
