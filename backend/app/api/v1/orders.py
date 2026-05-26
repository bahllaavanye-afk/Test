"""Order submission and management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.order import Order
from app.models.user import User
from pydantic import BaseModel, field_validator, Field
from datetime import datetime, timezone
import uuid

router = APIRouter(prefix="/orders", tags=["orders"])


class OrderCreate(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20, pattern=r"^[A-Za-z0-9:/._-]+$")
    side: str = Field(..., pattern=r"^(buy|sell)$")
    order_type: str = Field("market", pattern=r"^(market|limit|stop|stop_limit)$")
    quantity: float = Field(..., gt=0, le=1_000_000)
    limit_price: float | None = Field(None, gt=0)
    stop_price: float | None = Field(None, gt=0)
    execution_algo: str = Field("auto", pattern=r"^(auto|market|limit_first|twap|vwap|iceberg)$")
    account_id: str = Field(..., min_length=1, max_length=64)

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.upper()


class OrderOut(BaseModel):
    id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    limit_price: float | None
    status: str
    filled_qty: float
    execution_algo: str | None
    created_at: datetime

    class Config:
        from_attributes = True


@router.get("/", response_model=list[OrderOut])
async def list_orders(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Order).order_by(Order.created_at.desc()).limit(limit)
    )
    return result.scalars().all()


@router.post("/", response_model=OrderOut)
async def submit_order(
    body: OrderCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    order = Order(
        id=str(uuid.uuid4()),
        account_id=body.account_id,
        symbol=body.symbol,
        side=body.side,
        order_type=body.order_type,
        quantity=body.quantity,
        limit_price=body.limit_price,
        stop_price=body.stop_price,
        execution_algo=body.execution_algo,
        status="pending",
        filled_qty=0.0,
        submitted_at=datetime.now(timezone.utc),
    )
    db.add(order)
    await db.commit()
    await db.refresh(order)
    return order


@router.delete("/{order_id}")
async def cancel_order(
    order_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Order).where(Order.id == order_id))
    order = result.scalar_one_or_none()
    if not order:
        raise HTTPException(404, "Order not found")
    order.status = "cancelled"
    await db.commit()
    return {"cancelled": order_id}
