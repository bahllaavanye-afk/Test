"""Order submission and management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.order import Order
from app.models.user import User
from app.models.account import Account
from app.models.audit_log import AuditLog
from pydantic import BaseModel, ConfigDict, field_validator, Field, model_validator
from datetime import datetime, timezone
import uuid
from app.utils.logging import logger

router = APIRouter(prefix="/orders", tags=["orders"])


class OrderCreate(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20, pattern=r"^[A-Za-z0-9:/._-]+$")
    side: str = Field(..., pattern=r"^(buy|sell)$")
    order_type: str = Field("market", pattern=r"^(market|limit|stop|stop_limit)$")
    quantity: float | None = Field(None, gt=0, le=1_000_000)  # make optional if notional given
    notional: float | None = Field(None, gt=0)  # dollar amount instead of shares
    limit_price: float | None = Field(None, gt=0)
    stop_price: float | None = Field(None, gt=0)
    take_profit_price: float | None = Field(None, gt=0)
    stop_loss_price: float | None = Field(None, gt=0)
    trailing_stop_pct: float | None = Field(None, gt=0, le=50)  # max 50% trailing
    time_in_force: str = Field("gtc", pattern=r"^(gtc|day|ioc|fok|opg|cls)$")
    execution_algo: str = Field("auto", pattern=r"^(auto|market|limit_first|twap|vwap|iceberg)$")
    account_id: str = Field(..., min_length=1, max_length=64)

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def require_qty_or_notional(self):
        if self.quantity is None and self.notional is None:
            raise ValueError("Either quantity or notional must be provided")
        return self


class OrderOut(BaseModel):
    id: str
    symbol: str
    side: str
    order_type: str
    quantity: float | None
    limit_price: float | None
    stop_price: float | None
    stop_loss_price: float | None
    take_profit_price: float | None
    trailing_stop_pct: float | None
    notional: float | None
    time_in_force: str
    bracket_parent_id: str | None
    status: str
    filled_qty: float
    execution_algo: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class OrderModify(BaseModel):
    limit_price: float | None = Field(None, gt=0)
    stop_price: float | None = Field(None, gt=0)
    take_profit_price: float | None = Field(None, gt=0)
    stop_loss_price: float | None = Field(None, gt=0)
    trailing_stop_pct: float | None = Field(None, gt=0, le=50)
    quantity: float | None = Field(None, gt=0)


async def _log_audit(
    db: AsyncSession,
    user_id: str,
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    extra_data: dict | None = None,
    request: Request | None = None,
):
    """Write an audit log entry."""
    ip_address = None
    user_agent = None
    if request is not None:
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent", "")[:256]
    log = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip_address,
        user_agent=user_agent,
        extra_data=extra_data or {},
    )
    db.add(log)
    # Do not commit here — caller commits with the surrounding transaction


@router.get("/", response_model=list[OrderOut])
async def list_orders(
    limit: int = Query(default=50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Order)
        .join(Account, Order.account_id == Account.id)
        .where(Account.user_id == current_user.id)
        .order_by(Order.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


@router.post("/", response_model=OrderOut)
async def submit_order(
    body: OrderCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    request: Request = None,
):
    acct_result = await db.execute(select(Account).where(Account.id == body.account_id))
    account = acct_result.scalar_one_or_none()
    if not account or account.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    order = Order(
        id=str(uuid.uuid4()),
        account_id=body.account_id,
        symbol=body.symbol,
        side=body.side,
        order_type=body.order_type,
        quantity=body.quantity,
        notional=body.notional,
        limit_price=body.limit_price,
        stop_price=body.stop_price,
        take_profit_price=body.take_profit_price,
        stop_loss_price=body.stop_loss_price,
        trailing_stop_pct=body.trailing_stop_pct,
        time_in_force=body.time_in_force.upper(),
        execution_algo=body.execution_algo,
        status="pending",
        filled_qty=0.0,
        submitted_at=datetime.now(timezone.utc),
    )
    db.add(order)

    # Audit log
    await _log_audit(
        db, current_user.id, "order_submit",
        resource_type="order", resource_id=order.id,
        extra_data={"symbol": body.symbol, "side": body.side, "order_type": body.order_type},
        request=request,
    )

    await db.commit()
    await db.refresh(order)

    # Try to route to broker if account has broker credentials
    from app.brokers.alpaca_orders import submit_alpaca_order
    if account.broker == "alpaca" and account.encrypted_key:
        try:
            alpaca_resp = await submit_alpaca_order(account, {
                "symbol": order.symbol,
                "side": order.side,
                "order_type": order.order_type,
                "quantity": order.quantity,
                "notional": body.notional,
                "limit_price": order.limit_price,
                "stop_price": order.stop_price,
                "take_profit_price": body.take_profit_price,
                "stop_loss_price": body.stop_loss_price,
                "trailing_stop_pct": body.trailing_stop_pct,
                "time_in_force": body.time_in_force,
            })
            order.broker_order_id = alpaca_resp.get("id")
            order.status = alpaca_resp.get("status", "submitted")
            order.raw_payload = alpaca_resp
            await db.commit()
            await db.refresh(order)
        except Exception as e:
            logger.warning(f"Alpaca order submission failed: {e} — order saved locally only")

    return order


@router.post("/bracket", response_model=list[OrderOut])
async def submit_bracket(
    body: OrderCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    request: Request = None,
):
    """Create a parent order plus take-profit and/or stop-loss child orders atomically."""
    acct_result = await db.execute(select(Account).where(Account.id == body.account_id))
    account = acct_result.scalar_one_or_none()
    if not account or account.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    child_side = "sell" if body.side == "buy" else "buy"

    # Parent order
    parent = Order(
        id=str(uuid.uuid4()),
        account_id=body.account_id,
        symbol=body.symbol,
        side=body.side,
        order_type=body.order_type,
        quantity=body.quantity,
        notional=body.notional,
        limit_price=body.limit_price,
        stop_price=body.stop_price,
        take_profit_price=body.take_profit_price,
        stop_loss_price=body.stop_loss_price,
        trailing_stop_pct=body.trailing_stop_pct,
        time_in_force=body.time_in_force.upper(),
        execution_algo=body.execution_algo,
        status="pending",
        filled_qty=0.0,
        submitted_at=datetime.now(timezone.utc),
    )
    db.add(parent)

    created_orders = [parent]

    # Take-profit child (LIMIT order on opposite side)
    if body.take_profit_price:
        tp_order = Order(
            id=str(uuid.uuid4()),
            account_id=body.account_id,
            symbol=body.symbol,
            side=child_side,
            order_type="limit",
            quantity=body.quantity,
            notional=body.notional,
            limit_price=body.take_profit_price,
            time_in_force=body.time_in_force.upper(),
            status="bracket_pending",
            filled_qty=0.0,
            bracket_parent_id=parent.id,
        )
        db.add(tp_order)
        created_orders.append(tp_order)

    # Stop-loss child (STOP order on opposite side)
    if body.stop_loss_price:
        sl_order = Order(
            id=str(uuid.uuid4()),
            account_id=body.account_id,
            symbol=body.symbol,
            side=child_side,
            order_type="stop",
            quantity=body.quantity,
            notional=body.notional,
            stop_price=body.stop_loss_price,
            trailing_stop_pct=body.trailing_stop_pct,
            time_in_force=body.time_in_force.upper(),
            status="bracket_pending",
            filled_qty=0.0,
            bracket_parent_id=parent.id,
        )
        db.add(sl_order)
        created_orders.append(sl_order)

    # Audit log for bracket submission
    await _log_audit(
        db, current_user.id, "order_submit",
        resource_type="order", resource_id=parent.id,
        extra_data={
            "symbol": body.symbol, "side": body.side,
            "order_type": body.order_type, "bracket": True,
            "legs": len(created_orders),
        },
        request=request,
    )

    await db.commit()

    # Try to route the bracket to Alpaca (Alpaca supports bracket natively)
    from app.brokers.alpaca_orders import submit_alpaca_order
    if account.broker == "alpaca" and account.encrypted_key:
        try:
            alpaca_resp = await submit_alpaca_order(account, {
                "symbol": parent.symbol,
                "side": parent.side,
                "order_type": parent.order_type,
                "quantity": parent.quantity,
                "notional": body.notional,
                "limit_price": parent.limit_price,
                "stop_price": parent.stop_price,
                "take_profit_price": body.take_profit_price,
                "stop_loss_price": body.stop_loss_price,
                "trailing_stop_pct": body.trailing_stop_pct,
                "time_in_force": body.time_in_force,
            })
            parent.broker_order_id = alpaca_resp.get("id")
            parent.status = alpaca_resp.get("status", "submitted")
            parent.raw_payload = alpaca_resp
            await db.commit()
        except Exception as e:
            logger.warning(f"Alpaca bracket order submission failed: {e} — orders saved locally only")

    for o in created_orders:
        await db.refresh(o)

    return created_orders


@router.patch("/{order_id}", response_model=OrderOut)
async def modify_order(
    order_id: str,
    body: OrderModify,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    request: Request = None,
):
    """Modify an open order's price, quantity, or bracket legs."""
    result = await db.execute(select(Order).where(Order.id == order_id))
    order = result.scalar_one_or_none()
    if not order:
        raise HTTPException(404, "Order not found")

    # Verify ownership via account
    acct_result = await db.execute(select(Account).where(Account.id == order.account_id))
    account = acct_result.scalar_one_or_none()
    if not account or account.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Only allow modification for modifiable statuses
    if order.status not in ("pending", "open", "partially_filled"):
        raise HTTPException(400, f"Cannot modify order with status '{order.status}'")

    changes = {}
    # Update only provided (non-None) fields
    if body.limit_price is not None:
        order.limit_price = body.limit_price
        changes["limit_price"] = body.limit_price
    if body.stop_price is not None:
        order.stop_price = body.stop_price
        changes["stop_price"] = body.stop_price
    if body.take_profit_price is not None:
        order.take_profit_price = body.take_profit_price
        changes["take_profit_price"] = body.take_profit_price
    if body.stop_loss_price is not None:
        order.stop_loss_price = body.stop_loss_price
        changes["stop_loss_price"] = body.stop_loss_price
    if body.trailing_stop_pct is not None:
        order.trailing_stop_pct = body.trailing_stop_pct
        changes["trailing_stop_pct"] = body.trailing_stop_pct
    if body.quantity is not None:
        order.quantity = body.quantity
        changes["quantity"] = body.quantity

    # Audit log
    await _log_audit(
        db, current_user.id, "order_modify",
        resource_type="order", resource_id=order.id,
        extra_data={"changes": changes},
        request=request,
    )

    await db.commit()

    # Propagate changes to Alpaca if applicable
    if account.broker == "alpaca" and account.encrypted_key and order.broker_order_id and changes:
        from app.brokers.alpaca_orders import modify_alpaca_order
        try:
            await modify_alpaca_order(account, order.broker_order_id, changes)
        except Exception as e:
            logger.warning(f"Alpaca order modification failed: {e} — local update applied only")

    await db.refresh(order)
    return order


@router.delete("/{order_id}")
async def cancel_order(
    order_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    request: Request = None,
):
    result = await db.execute(select(Order).where(Order.id == order_id))
    order = result.scalar_one_or_none()
    if not order:
        raise HTTPException(404, "Order not found")
    acct_result = await db.execute(select(Account).where(Account.id == order.account_id))
    account = acct_result.scalar_one_or_none()
    if not account or account.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    order.status = "cancelled"
    order.cancelled_at = datetime.now(timezone.utc)

    # Cancel on Alpaca broker if applicable
    if account.broker == "alpaca" and account.encrypted_key and order.broker_order_id:
        from app.brokers.alpaca_orders import cancel_alpaca_order
        try:
            await cancel_alpaca_order(account, order.broker_order_id)
        except Exception as e:
            logger.warning(f"Alpaca order cancel failed: {e} — local cancel applied only")

    await _log_audit(
        db, current_user.id, "order_cancel",
        resource_type="order", resource_id=order.id,
        extra_data={"symbol": order.symbol},
        request=request,
    )

    await db.commit()
    return {"cancelled": order_id}


@router.get("/{order_id}/bracket-legs", response_model=list[OrderOut])
async def get_bracket_legs(
    order_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return child orders linked to a parent order via bracket_parent_id."""
    # Verify parent exists and belongs to current user
    result = await db.execute(select(Order).where(Order.id == order_id))
    parent = result.scalar_one_or_none()
    if not parent:
        raise HTTPException(404, "Order not found")
    acct_result = await db.execute(select(Account).where(Account.id == parent.account_id))
    account = acct_result.scalar_one_or_none()
    if not account or account.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    legs_result = await db.execute(
        select(Order).where(Order.bracket_parent_id == order_id)
    )
    return legs_result.scalars().all()
