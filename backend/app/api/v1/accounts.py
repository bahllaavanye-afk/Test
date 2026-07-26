"""Account management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.account import Account
from app.models.audit_log import AuditLog
from app.models.user import User
from app.utils.security import encrypt_secret
from app.utils.logging import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

router = APIRouter(prefix="/accounts", tags=["accounts"])


async def latest_total_equity(db: AsyncSession) -> float:
    """Sum of each active account's most recent snapshot equity.

    The Account model itself has NO equity column — equity is a time series on
    AccountSnapshot (written hourly from live broker data). Every caller that
    wants "current equity" must read the latest snapshot, not the account row;
    reading `account.total_equity` is an AttributeError.
    """
    from app.models.account import AccountSnapshot

    account_ids = (
        (await db.execute(select(Account.id).where(Account.is_active == True)))  # noqa: E712
        .scalars()
        .all()
    )
    total = 0.0
    for acc_id in account_ids:
        snap = (
            await db.execute(
                select(AccountSnapshot.total_equity)
                .where(AccountSnapshot.account_id == acc_id)
                .order_by(AccountSnapshot.ts.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        total += float(snap or 0)
    return total


class AccountCreate(BaseModel):
    broker: str = Field(..., description="Broker name, e.g., 'alpaca'", example="alpaca")
    label: str = Field(..., description="User-friendly label for the account", example="My Alpaca Paper Account")
    mode: str = Field("paper", description="Trading mode, either 'paper' or 'live'", example="paper")
    api_key: str = Field(..., description="API key for broker authentication", example="AK123456789")
    api_secret: str = Field(..., description="API secret for broker authentication", example="secret123")
    extra_config: dict = Field(
        default_factory=dict,
        description="Optional additional configuration for the broker",
        example={"region": "us-east-1"},
    )

    @field_validator("mode")
    def validate_mode(cls, v: str) -> str:
        if v not in {"paper", "live"}:
            raise ValueError("mode must be either 'paper' or 'live'")
        return v

    @field_validator("broker")
    def validate_broker(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("broker must be a non-empty string")
        return v


class AccountOut(BaseModel):
    id: str = Field(..., description="Unique account identifier", example="a1b2c3d4")
    broker: str = Field(..., description="Broker name", example="alpaca")
    label: str = Field(..., description="User-defined label", example="My Alpaca Paper Account")
    mode: str = Field(..., description="Trading mode", example="paper")
    extra_config: dict = Field(
        default_factory=dict,
        description="Additional broker configuration",
        example={"region": "us-east-1"},
    )

    model_config = ConfigDict(from_attributes=True)


class AccountEquityOut(BaseModel):
    equity: float = Field(..., description="Total equity value", example=10000.0, ge=0)
    cash: float = Field(..., description="Cash balance", example=5000.0, ge=0)
    buying_power: float = Field(..., description="Available buying power", example=15000.0, ge=0)
    portfolio_value: float = Field(..., description="Total portfolio value", example=12000.0, ge=0)
    day_trade_count: int | None = Field(
        None,
        description="Number of day trades today",
        example=2,
        ge=0,
    )
    pattern_day_trader: bool | None = Field(
        None,
        description="Whether the account is flagged as a pattern day trader",
        example=False,
    )


@router.get("/", response_model=list[AccountOut])
async def list_accounts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Account).where(Account.user_id == current_user.id))
    return result.scalars().all()


@router.post("/", response_model=AccountOut)
async def create_account(
    body: AccountCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    request: Request = None,
):
    account = Account(
        user_id=current_user.id,
        broker=body.broker,
        label=body.label,
        mode=body.mode,
        encrypted_key=encrypt_secret(body.api_key),
        encrypted_secret=encrypt_secret(body.api_secret),
        extra_config=body.extra_config,
    )
    db.add(account)

    # Audit log for key addition
    log = AuditLog(
        user_id=current_user.id,
        action="key_add",
        resource_type="account",
        resource_id=None,  # will be set after commit
        ip_address=request.client.host if (request and request.client) else None,
        user_agent=(request.headers.get("user-agent", "")[:256] if request else None),
        extra_data={"broker": body.broker, "mode": body.mode},
    )
    db.add(log)

    await db.commit()
    await db.refresh(account)

    # Update the audit log with the new account id
    log.resource_id = account.id
    await db.commit()

    return account


@router.get("/{account_id}/equity", response_model=AccountEquityOut)
async def get_account_equity(
    account_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return live equity, buying power, and day-trade count from Alpaca."""
    result = await db.execute(
        select(Account).where(Account.id == account_id, Account.user_id == current_user.id)
    )
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(404, "Account not found")

    if account.broker != "alpaca" or not account.encrypted_key:
        raise HTTPException(400, "Live equity is only available for Alpaca accounts with stored credentials")

    from app.brokers.alpaca_orders import get_alpaca_account

    try:
        data = await get_alpaca_account(account)
    except Exception as e:
        logger.warning(f"Alpaca account fetch failed for account {account_id}: {e}")
        raise HTTPException(502, "Unable to fetch live account data from Alpaca")

    return AccountEquityOut(
        equity=float(data.get("equity", 0)),
        cash=float(data.get("cash", 0)),
        buying_power=float(data.get("buying_power", 0)),
        portfolio_value=float(data.get("portfolio_value", 0)),
        day_trade_count=int(data["daytrade_count"]) if data.get("daytrade_count") is not None else None,
        pattern_day_trader=bool(data.get("pattern_day_trader")) if data.get("pattern_day_trader") is not None else None,
    )


@router.delete("/{account_id}")
async def delete_account(
    account_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Account).where(Account.id == account_id, Account.user_id == current_user.id))
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(404, "Account not found")
    await db.delete(account)
    await db.commit()
    return {"deleted": account_id}