"""Portfolio positions endpoint."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.position import Position
from app.models.user import User
from app.models.account import Account
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(prefix="/positions", tags=["positions"])


class PositionOut(BaseModel):
    id: str
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float | None
    unrealized_pnl: float | None
    side: str

    class Config:
        from_attributes = True


@router.get("/", response_model=list[PositionOut])
async def list_positions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Position)
        .join(Account, Position.account_id == Account.id)
        .where(Account.user_id == current_user.id)
        .where(Position.quantity != 0)
    )
    return result.scalars().all()
