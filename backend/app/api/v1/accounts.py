"""Account management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.database import get_db
from app.api.deps import get_current_user
from app.models.account import Account
from app.models.audit_log import AuditLog
from app.models.user import User
from app.utils.security import encrypt_secret
from app.utils.logging import logger
from pydantic import BaseModel, ConfigDict, Field
import asyncio

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
    broker: str
    label: str
    mode: str = "paper"
    api_key: str
    api_secret: str
    extra_config: dict = Field(default_factory=dict)


class AccountOut(BaseModel):
    id: str
    broker: str
    label: str
    mode: str
    extra_config: dict

    model_config = ConfigDict(from_attributes=True)


class AccountEquityOut(BaseModel):
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    day_trade_count: int | None
    pattern_day_trader: bool | None


def _validate_equity_data(data: dict) -> None:
    """Basic sanity checks for equity related fields."""
    for key in ("equity", "cash", "buying_power", "portfolio_value"):
        value = data.get(key)
        if value is not None and float(value) < 0:
            raise ValueError(f"{key} cannot be negative: {value}")


async def _fetch_alpaca_account_with_retry(account: Account, retries: int = 3, delay: float = 0.5):
    """Retry wrapper for Alpaca account fetch with exponential backoff."""
    from app.brokers.alpaca_orders import get_alpaca_account

    attempt = 0
    while True:
        try:
            data = await get_alpaca_account(account)
            _validate_equity_data(data)
            return data
        except Exception as exc:  # pylint: disable=broad-except
            attempt += 1
            if attempt > retries:
                logger.error(
                    f"Alpaca account fetch failed after {retries} retries for account {account.id}: {exc}"
                )
                raise
            backoff = delay * (2 ** (attempt - 1))
            logger.warning(
                f"Alpaca fetch attempt {attempt} failed for account {account.id}: {exc}. Retrying in {backoff}s."
            )
            await asyncio.sleep(backoff)


@router.get("/", response_model=list[AccountOut])
async def list_accounts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return only active accounts belonging to the current user."""
    stmt = select(Account).where(
        and_(Account.user_id == current_user.id, Account.is_active == True)  # noqa: E712
    )
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("/", response_model=AccountOut)
async def create_account(
    body: AccountCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    request: Request = None,
):
    """Create a new account with encrypted credentials and audit logging."""
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
    """Return live equity, buying power, and day‑trade count from Alpaca.

    Includes confirmation filters and retry logic to improve data reliability.
    """
    stmt = select(Account).where(
        and_(Account.id == account_id, Account.user_id == current_user.id)
    )
    result = await db.execute(stmt)
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(404, "Account not found")

    if not account.is_active:
        raise HTTPException(400, "Account is inactive")

    if account.broker != "alpaca" or not account.encrypted_key:
        raise HTTPException(
            400,
            "Live equity is only available for Alpaca accounts with stored credentials",
        )

    try:
        data = await _fetch_alpaca_account_with_retry(account)
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
    """Delete an account after ensuring no open positions exist."""
    stmt = select(Account).where(
        and_(Account.id == account_id, Account.user_id == current_user.id)
    )
    result = await db.execute(stmt)
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(404, "Account not found")

    # Prevent deletion if the account has active positions (conservative exit logic)
    if hasattr(account, "has_open_positions") and account.has_open_positions:
        raise HTTPException(
            400,
            "Cannot delete account with open positions. Close positions before deletion.",
        )

    await db.delete(account)
    await db.commit()
    return {"deleted": account_id}