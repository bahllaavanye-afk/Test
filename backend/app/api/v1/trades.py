"""Trade history endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.trade import Trade
from app.models.user import User
from pydantic import BaseModel, ConfigDict, ConfigDict
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
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Trade).order_by(Trade.opened_at.desc()).limit(limit)
    )
    return result.scalars().all()
