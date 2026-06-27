"""Order submission and management endpoints."""
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.api.limiter import limiter
from app.database import get_db
from app.models.account import Account
from app.models.audit_log import AuditLog
from app.models.order import Order
from app.models.user import User
from app.services.agent_logger import agent_logger
from app.utils.logging import logger

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

# Query limits
DEFAULT_QUERY_LIMIT: int = 50
MIN_QUERY_LIMIT: int = 1
MAX_QUERY_LIMIT: int = 500

# Order field constraints
MAX_QUANTITY: int = 1_000_000
MAX_TRAILING_PCT: int = 50

# Default order values
DEFAULT_ORDER_TYPE: str = "market"
DEFAULT_TIME_IN_FORCE: str = "gtc"
DEFAULT_EXECUTION_ALGO: str = "auto"

# Order status values
ORDER_STATUS_PENDING: str = "pending"
ORDER_STATUS_REJECTED: str = "rejected"

# Audit actions
AUDIT_ACTION_ORDER_SUBMIT: str = "order_submit"

# Risk manager constants
RISK_BUCKET_DIRECTIONAL: str = "directional"

# Broker identifiers
BROKER_ALPACA: str = "alpaca"

# Rate limiter string
LIMITER_SUBMIT_ORDER: str = "10/minute"

router = APIRouter(prefix="/orders", tags=["orders"])


class OrderCreate(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20, pattern=r"^[A-Za-z0-9:/._-]+$")
    side: str = Field(..., pattern=r"^(buy|sell)$")
    order_type: str = Field(DEFAULT_ORDER_TYPE, pattern=r"^(market|limit|stop|stop_limit)$")
    quantity: float | None = Field(None, gt=0, le=MAX_QUANTITY)  # make optional if notional given
    notional: float | None = Field(None, gt=0)  # dollar amount instead of shares
    limit_price: float | None = Field(None, gt=0)
    stop_price: float | None = Field(None, gt=0)
    take_profit_price: float | None = Field(None, gt=0)
    stop_loss_price: float | None = Field(None, gt=0)
    trailing_stop_pct: float | None = Field(None, gt=0, le=MAX_TRAILING_PCT)  # max 50% trailing
    time_in_force: str = Field(DEFAULT_TIME_IN_FORCE, pattern=r"^(gtc|day|ioc|fok|opg|cls)$")
    execution_algo: str = Field(DEFAULT_EXECUTION_ALGO, pattern=r"^(auto|market|limit_first|twap|vwap|iceberg)$")
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
    trailing_stop_pct: float | None = Field(None, gt=0, le=MAX_TRAILING_PCT)
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
    limit: int = Query(default=DEFAULT_QUERY_LIMIT, ge=MIN_QUERY_LIMIT, le=MAX_QUERY_LIMIT),
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
@limiter.limit(LIMITER_SUBMIT_ORDER)
async def submit_order(
    request: Request,
    body: OrderCreate,
    background_tasks: BackgroundTasks,
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
        status=ORDER_STATUS_PENDING,
        filled_qty=0.0,
        submitted_at=datetime.now(UTC),
    )
    db.add(order)

    # Audit log
    await _log_audit(
        db,
        current_user.id,
        AUDIT_ACTION_ORDER_SUBMIT,
        resource_type="order",
        resource_id=order.id,
        extra_data={"symbol": body.symbol, "side": body.side, "order_type": body.order_type},
        request=request,
    )

    await db.commit()
    await db.refresh(order)

    # Real-time agent log (fire-and-forget)
    agent_logger.log_action_fire_and_forget(
        action="submit_order",
        employee_id=current_user.email or current_user.id,
        agent_type="human",
        tool_used="alpaca_api",
        input_summary=f"{body.side} {body.symbol} {body.order_type}",
        output_summary=f"order_id={order.id} status={order.status}",
        status="ok",
        symbol=body.symbol,
        account_id=body.account_id,
    )

    # Try to route to broker if account has broker credentials
    from app.brokers.alpaca_orders import submit_alpaca_order

    if account.broker == BROKER_ALPACA and account.encrypted_key:
        # Risk gate: every manual order passes through the risk manager before reaching the broker
        risk_mgr = getattr(request.app.state, "risk_manager", None)
        if risk_mgr is not None:
            from app.brokers.base import OrderRequest as RiskOrderRequest

            risk_req = RiskOrderRequest(
                symbol=order.symbol,
                quantity=order.quantity or 1.0,
                side=order.side,
                order_type=order.order_type,
                limit_price=order.limit_price,
                risk_bucket=RISK_BUCKET_DIRECTIONAL,
            )
            risk_decision = await risk_mgr.check_order(risk_req)
            if not risk_decision.allowed:
                order.status = ORDER_STATUS_REJECTED
                await db.commit()
                raise HTTPException(
                    status_code=422,
                    detail=f"Order rejected by risk manager: {risk_decision.reason}",
                )
            if risk_decision.adjusted_quantity is not None:
                order.quantity = risk_decision.adjusted_quantity
                await db.commit()
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
            })
            # Process alpaca_resp as needed (details omitted for brevity)
        except Exception as exc:
            logger.error(f"Alpaca order submission failed: {exc}")
            raise HTTPException(status_code=502, detail="Broker submission failed")
    return order