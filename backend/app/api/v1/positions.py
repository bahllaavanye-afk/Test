"""Portfolio positions endpoint."""
from fastapi import APIRouter, Depends, Query
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
