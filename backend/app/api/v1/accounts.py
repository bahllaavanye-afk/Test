"""Account management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.account import Account
from app.models.audit_log import AuditLog
from app.models.user import User
from app.utils.security import encrypt_secret, decrypt_secret
from app.utils.logging import logger
from pydantic import BaseModel, ConfigDict, Field, validator


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
    broker: str = Field(..., description="Broker identifier (e.g., 'alpaca')", example="alpaca")
    label: str = Field(..., description="Human‑readable label for the account", example="My Alpaca Paper Account")
    mode: str = Field(
        "paper",
        description="Execution mode, either 'paper' for simulated trading or 'live' for real trading",
        example="paper",
    )
    api_key: str = Field(..., description="API key provided by the broker", example="AK1234567890")
    api_secret: str = Field(..., description="API secret provided by the broker", example="secret123")
    extra_config: dict = Field(
        default_factory=dict,
        description="Optional broker‑specific configuration parameters",
        example={"region": "us-east-1"},
    )

    @validator("mode")
    def validate_mode(cls, v: str) -> str:
        allowed = {"paper", "live"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v

    @validator("broker", "api_key", "api_secret")
    def non_empty_strings(cls, v: str, field) -> str:
        if not v or not v.strip():
            raise ValueError(f"{field.name} must be a non‑empty string")
        return v.strip()

    @validator("extra_config")
    def validate_extra_config(cls, v: dict) -> dict:
        if not isinstance(v, dict):
            raise ValueError("extra_config must be a dictionary")
        return v


class AccountOut(BaseModel):
    id: str = Field(..., description="Unique identifier of the account", example="a1b2c3d4")
    broker: str = Field(..., description="Broker identifier", example="alpaca")
    label: str = Field(..., description="Human‑readable label", example="My Alpaca Paper Account")
    mode: str = Field(..., description="Execution mode ('paper' or 'live')", example="paper")
    extra_config: dict = Field(
        default_factory=dict,
        description="Broker‑specific configuration parameters",
        example={"region": "us-east-1"},
    )

    model_config = ConfigDict(from_attributes=True)


class AccountEquityOut(BaseModel):
    equity: float = Field(..., description="Total equity value of the account", example=10500.75)
    cash: float = Field(..., description="Cash balance available", example=2500.00)
    buying_power: float = Field(..., description="Available buying power", example=8000.00)
    portfolio_value: float = Field(..., description="Current value of the portfolio holdings", example=8000.75)
    day_trade_count: int | None = Field(
        None,
        description="Number of day trades executed today, if applicable",
        example=2,
    )
    pattern_day_trader: bool | None = Field(
        None,
        description="Flag indicating if the account is marked as a pattern day trader",
        example=False,
    )

    model_config = ConfigDict(from_attributes=True)


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
    """Return live equity, buying power, and day‑trade count from Alpaca."""
    result = await db.execute(
        select(Account).where(Account.id == account_id, Account.user_id == current_user.id)
    )
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(404, "Account not found")

    if account.broker != "alpaca" or not account.encrypted_key:
        raise HTTPException(
            400,
            "Live equity is only available for Alpaca accounts with stored credentials",
        )

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
        day_trade_count=int(data["daytrade_count"])
        if data.get("daytrade_count") is not None
        else None,
        pattern_day_trader=bool(data.get("pattern_day_trader"))
        if data.get("pattern_day_trader") is not None
        else None,
    )


@router.delete("/{account_id}")
async def delete_account(
    account_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Account).where(Account.id == account_id, Account.user_id == current_user.id)
    )
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(404, "Account not found")
    await db.delete(account)
    await db.commit()
    return {"deleted": account_id}