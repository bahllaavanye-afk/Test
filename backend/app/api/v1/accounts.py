"""Account management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import get_db
from app.api.deps import get_current_user
from app.models.account import Account
from app.models.audit_log import AuditLog
from app.models.trade import Trade
from app.models.user import User
from app.utils.security import encrypt_secret, decrypt_secret
from app.utils.logging import logger
from pydantic import BaseModel, ConfigDict
from datetime import datetime, timezone, timedelta
import pandas as pd

router = APIRouter(prefix="/accounts", tags=["accounts"])


class AccountCreate(BaseModel):
    broker: str
    label: str
    mode: str = "paper"
    api_key: str
    api_secret: str
    extra_config: dict = {}


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


_LIVE_PROMOTION_MIN_SHARPE = 1.5
_LIVE_PROMOTION_MIN_TRADES = 30
_LIVE_PROMOTION_MIN_DAYS = 14


@router.post("/{account_id}/promote-to-live")
async def promote_to_live(
    account_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    request: Request = None,
):
    """
    Promote a paper trading account to live mode if it passes the Sharpe gate.

    Requirements (all must pass):
      - ≥ 30 closed trades in the last 14 days
      - Annualised Sharpe ≥ 1.5 over those trades
      - Account is currently in 'paper' mode

    Returns the updated account on success, or a 422 with the reason it failed.
    """
    result = await db.execute(
        select(Account).where(Account.id == account_id, Account.user_id == current_user.id)
    )
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(404, "Account not found")

    if account.mode != "paper":
        raise HTTPException(400, f"Account is already in '{account.mode}' mode")

    # Fetch last 14 days of trades for this account
    since = datetime.now(timezone.utc) - timedelta(days=_LIVE_PROMOTION_MIN_DAYS)
    trades_result = await db.execute(
        select(
            func.date_trunc("day", Trade.closed_at).label("day"),
            func.sum(Trade.realized_pnl).label("daily_pnl"),
            func.count(Trade.id).label("trade_count"),
        )
        .where(
            Trade.account_id == account_id,
            Trade.closed_at >= since,
            Trade.realized_pnl.isnot(None),
        )
        .group_by(func.date_trunc("day", Trade.closed_at))
        .order_by(func.date_trunc("day", Trade.closed_at))
    )
    rows = trades_result.all()

    total_trades = int(sum(r.trade_count for r in rows))
    if total_trades < _LIVE_PROMOTION_MIN_TRADES:
        raise HTTPException(
            422,
            f"Insufficient trades: {total_trades} in last {_LIVE_PROMOTION_MIN_DAYS} days "
            f"(need ≥ {_LIVE_PROMOTION_MIN_TRADES})",
        )

    # Compute annualised Sharpe from daily P&L
    daily_pnls = [float(r.daily_pnl) for r in rows]
    s = pd.Series(daily_pnls)
    std = float(s.std())
    if std <= 0 or len(s) < 5:
        raise HTTPException(422, "Insufficient daily P&L variance to compute Sharpe — need ≥ 5 trading days")

    sharpe = float(s.mean() / std * (252 ** 0.5))
    if sharpe < _LIVE_PROMOTION_MIN_SHARPE:
        raise HTTPException(
            422,
            f"Sharpe too low: {sharpe:.2f} (need ≥ {_LIVE_PROMOTION_MIN_SHARPE}). "
            f"Keep running in paper mode.",
        )

    # All gates passed — promote to live
    account.mode = "live"
    log = AuditLog(
        user_id=current_user.id,
        action="promote_to_live",
        resource_type="account",
        resource_id=account_id,
        ip_address=request.client.host if (request and request.client) else None,
        user_agent=(request.headers.get("user-agent", "")[:256] if request else None),
        extra_data={
            "sharpe": round(sharpe, 4),
            "total_trades": total_trades,
            "days_evaluated": len(rows),
            "min_sharpe_required": _LIVE_PROMOTION_MIN_SHARPE,
        },
    )
    db.add(log)
    await db.commit()
    await db.refresh(account)

    logger.info(
        "account.promoted_to_live",
        account_id=account_id,
        sharpe=round(sharpe, 4),
        total_trades=total_trades,
    )
    return {
        "account": AccountOut.model_validate(account),
        "promotion_metrics": {
            "sharpe": round(sharpe, 4),
            "total_trades": total_trades,
            "days_evaluated": len(rows),
        },
        "message": f"Account promoted to live. Sharpe {sharpe:.2f} ≥ {_LIVE_PROMOTION_MIN_SHARPE} ✓",
    }


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
