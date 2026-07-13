"""Order submission and management endpoints."""
from datetime import datetime, timezone
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.api.limiter import limiter
from app.database import get_db
from app.models.account import Account
from app.models.audit_log import AuditLog
from app.models.order import Order
from app.models.user import User
from app.utils.logging import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


router = APIRouter(prefix="/orders", tags=["orders"])


class OrderCreate(BaseModel):
    # max_length 24 fits OCC option symbols (root ≤6 + YYMMDD + C/P + 8-digit strike)
    symbol: str = Field(..., min_length=1, max_length=24, pattern=r"^[A-Za-z0-9:/._-]+$")
    side: str = Field(..., pattern=r"^(buy|sell)$")
    order_type: str = Field("market", pattern=r"^(market|limit|stop|stop_limit)$")
    quantity: float | None = Field(None, gt=0, le=1_000_000)  # optional if notional given
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
) -> None:
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
    # Commit is deferred to the caller's transaction


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
@limiter.limit("10/minute")
async def submit_order(
    request: Request,
    body: OrderCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
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
        db,
        current_user.id,
        "order_submit",
        resource_type="order",
        resource_id=order.id,
        extra_data={
            "symbol": body.symbol,
            "side": body.side,
            "order_type": body.order_type,
        },
        request=request,
    )

    await db.commit()
    await db.refresh(order)

    # Route to broker if credentials are present
    if account.broker == "alpaca" and account.encrypted_key:
        # Risk gate: every manual order passes through the risk manager before the broker.
        risk_mgr = getattr(request.app.state, "risk_manager", None)
        if risk_mgr is not None:
            from app.brokers.base import OrderRequest as RiskOrderRequest

            risk_req = RiskOrderRequest(
                symbol=order.symbol,
                quantity=order.quantity or 1.0,
                side=order.side,
                order_type=order.order_type,
                limit_price=order.limit_price,
                risk_bucket="directional",
            )
            risk_decision = await risk_mgr.check_order(risk_req)
            if not risk_decision.allowed:
                order.status = "rejected"
                await db.commit()
                raise HTTPException(
                    status_code=422,
                    detail=f"Order rejected by risk manager: {risk_decision.reason}",
                )
            if risk_decision.adjusted_quantity is not None:
                order.quantity = risk_decision.adjusted_quantity
                await db.commit()

        from app.brokers.alpaca_orders import submit_alpaca_order

        try:
            alpaca_resp = await submit_alpaca_order(
                account,
                {
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
                },
            )
            order.broker_order_id = alpaca_resp.get("id")
            order.status = alpaca_resp.get("status", "submitted")
            order.raw_payload = alpaca_resp
            await db.commit()
            await db.refresh(order)
        except Exception as exc:  # pragma: no cover
            logger.error("Alpaca order submission failed: %s", exc, exc_info=True)
            order.status = "failed"
            await db.commit()
            raise HTTPException(status_code=502, detail="Broker submission failed") from exc

    return order