"""Account management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, DBAPIError
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
    """Sum of each active account's most recent snapshot equity.

    The Account model itself has NO equity column — equity is a time series on
    AccountSnapshot (written hourly from live broker data). Every caller that
    wants "current equity" must read the latest snapshot, not the account row;
    reading `account.total_equity` is an AttributeError.
    """
    from app.models.account import AccountSnapshot

    try:
        account_ids = (
            (await db.execute(select(Account.id).where(Account.is_active == True)))  # noqa: E712
            .scalars()
            .all()
        )
    except SQLAlchemyError as exc:
        logger.error(
            "Failed to fetch active account IDs for equity calculation",
            exc_info=exc,
            extra={"action": "latest_total_equity"},
        )
        raise HTTPException(status_code=500, detail="Database error while calculating equity")

    total = 0.0
    for acc_id in account_ids:
        try:
            snap = (
                await db.execute(
                    select(AccountSnapshot.total_equity)
                    .where(AccountSnapshot.account_id == acc_id)
                    .order_by(AccountSnapshot.ts.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        except SQLAlchemyError as exc:
            logger.error(
                "Failed to fetch latest snapshot for account",
                exc_info=exc,
                extra={"account_id": acc_id, "action": "latest_total_equity"},
            )
            continue
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


@router.get("/", response_model=list[AccountOut])
async def list_accounts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        result = await db.execute(select(Account).where(Account.user_id == current_user.id))
        return result.scalars().all()
    except SQLAlchemyError as exc:
        logger.error(
            "Error listing accounts",
            exc_info=exc,
            extra={"user_id": current_user.id, "action": "list_accounts"},
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve accounts")


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

    try:
        await db.commit()
        await db.refresh(account)
        # Update the audit log with the new account id
        log.resource_id = account.id
        await db.commit()
    except IntegrityError as exc:
        logger.error(
            "Integrity error creating account",
            exc_info=exc,
            extra={"user_id": current_user.id, "broker": body.broker, "action": "create_account"},
        )
        raise HTTPException(status_code=400, detail="Account creation conflict")
    except DBAPIError as exc:
        logger.error(
            "Database error creating account",
            exc_info=exc,
            extra={"user_id": current_user.id, "action": "create_account"},
        )
        raise HTTPException(status_code=500, detail="Database error during account creation")
    except SQLAlchemyError as exc:
        logger.error(
            "Unexpected error creating account",
            exc_info=exc,
            extra={"user_id": current_user.id, "action": "create_account"},
        )
        raise HTTPException(status_code=500, detail="Failed to create account")

    return account


@router.get("/{account_id}/equity", response_model=AccountEquityOut)
async def get_account_equity(
    account_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return live equity, buying power, and day-trade count from Alpaca."""
    try:
        result = await db.execute(
            select(Account).where(Account.id == account_id, Account.user_id == current_user.id)
        )
        account = result.scalar_one_or_none()
    except SQLAlchemyError as exc:
        logger.error(
            "Database error fetching account for equity",
            exc_info=exc,
            extra={"account_id": account_id, "user_id": current_user.id, "action": "get_account_equity"},
        )
        raise HTTPException(status_code=500, detail="Database error while retrieving account")

    if not account:
        raise HTTPException(404, "Account not found")

    if account.broker != "alpaca" or not account.encrypted_key:
        raise HTTPException(400, "Live equity is only available for Alpaca accounts with stored credentials")

    from app.brokers.alpaca_orders import get_alpaca_account

    try:
        data = await get_alpaca_account(account)
    except Exception as e:  # Specific broker errors could be caught if defined
        logger.error(
            "Alpaca account fetch failed",
            exc_info=e,
            extra={"account_id": account_id, "user_id": current_user.id, "action": "get_account_equity"},
        )
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
    try:
        result = await db.execute(select(Account).where(Account.id == account_id, Account.user_id == current_user.id))
        account = result.scalar_one_or_none()
    except SQLAlchemyError as exc:
        logger.error(
            "Database error locating account for deletion",
            exc_info=exc,
            extra={"account_id": account_id, "user_id": current_user.id, "action": "delete_account"},
        )
        raise HTTPException(status_code=500, detail="Database error while locating account")

    if not account:
        raise HTTPException(404, "Account not found")
    try:
        await db.delete(account)
        await db.commit()
    except SQLAlchemyError as exc:
        logger.error(
            "Database error deleting account",
            exc_info=exc,
            extra={"account_id": account_id, "user_id": current_user.id, "action": "delete_account"},
        )
        raise HTTPException(status_code=500, detail="Failed to delete account")
    return {"deleted": account_id}