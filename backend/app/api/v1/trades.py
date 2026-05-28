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
    quantity: float
    opened_at: datetime | None
    closed_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


@router.get("/", response_model=list[TradeOut])
async def list_trades(
    limit: int = Query(50, ge=1, le=500),
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
    result = await db.execute(query)
    return result.scalars().all()
