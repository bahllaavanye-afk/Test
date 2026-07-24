"""Account management endpoints."""

from typing import List, Dict, Optional

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
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/accounts", tags=["accounts"])


async def latest_total_equity(db: AsyncSession) -> float:
    """Calculate the sum of the most recent snapshot equity for all active accounts.

    The `Account` model does not contain an equity column; equity is stored in
    `AccountSnapshot` rows that are written hourly from live broker data. This
    function iterates over each active account, retrieves its latest snapshot,
    and aggregates the equity values.

    Args:
        db: An active asynchronous SQLAlchemy session.

    Returns:
        The total equity across all active accounts as a float.
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
    """Schema for creating a new trading account."""

    broker: str
    label: str
    mode: str = "paper"
    api_key: str
    api_secret: str
    extra_config: dict = Field(default_factory=dict)


class AccountOut(BaseModel):
    """Response model representing an account's public details."""

    id: str
    broker: str
    label: str
    mode: str
    extra_config: dict

    model_config = ConfigDict(from_attributes=True)


class AccountEquityOut(BaseModel):
    """Response model for live equity information of an account."""

    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    day_trade_count: Optional[int] = None
    pattern_day_trader: Optional[bool] = None


@router.get("/", response_model=List[AccountOut])
async def list_accounts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[AccountOut]:
    """List all accounts owned by the current user.

    Args:
        db: Asynchronous database session.
        current_user: The authenticated user requesting the accounts.

    Returns:
        A list of `AccountOut` objects representing the user's accounts.
    """
    result = await db.execute(select(Account).where(Account.user_id == current_user.id))
    return result.scalars().all()


@router.post("/", response_model=AccountOut)
async def create_account(
    body: AccountCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    request: Optional[Request] = None,
) -> AccountOut:
    """Create a new trading account and log the creation event.

    Args:
        body: The payload containing account creation details.
        db: Asynchronous database session.
        current_user: The authenticated user creating the account.
        request: The incoming HTTP request (optional, used for logging).

    Returns:
        The newly created `AccountOut` object.
    """
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
) -> AccountEquityOut:
    """Return live equity, buying power, and day‑trade count from Alpaca.

    Args:
        account_id: Identifier of the account to query.
        db: Asynchronous database session.
        current_user: The authenticated user requesting the data.

    Returns:
        An `AccountEquityOut` object containing the latest equity data.

    Raises:
        HTTPException: 404 if the account does not exist, 400 if the account is
            not an Alpaca account with stored credentials, or 502 if fetching
            data from Alpaca fails.
    """
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
) -> Dict[str, str]:
    """Delete an account owned by the current user.

    Args:
        account_id: Identifier of the account to delete.
        db: Asynchronous database session.
        current_user: The authenticated user performing the deletion.

    Returns:
        A dictionary confirming the deletion of the specified account.
    """
    result = await db.execute(select(Account).where(Account.id == account_id, Account.user_id == current_user.id))
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(404, "Account not found")
    await db.delete(account)
    await db.commit()
    return {"deleted": account_id}