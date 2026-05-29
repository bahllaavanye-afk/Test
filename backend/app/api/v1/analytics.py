"""Analytics and performance metrics endpoints."""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case
from datetime import datetime, timezone, timedelta, date
from typing import Optional
import re
import math
import httpx
import pandas as pd
from pydantic import BaseModel

from app.database import get_db
from app.api.deps import get_current_user
from app.models.trade import Trade
from app.models.slippage import SlippageRecord
from app.models.user import User
from app.models.account import Account
from app.models.position import Position
from app.models.order import Order
from app.config import settings

router = APIRouter(prefix="/analytics", tags=["analytics"])


async def _user_account_ids(db: AsyncSession, user_id: str) -> list[str]:
    """Return all account IDs owned by the given user. Used to scope queries."""
    result = await db.execute(
        select(Account.id).where(Account.user_id == user_id)
    )
    return [row[0] for row in result.all()]


@router.get("/performance")
async def get_performance(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Aggregate trade performance stats — scoped to current user's accounts."""
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return {"total_trades": 0, "avg_pnl": 0.0, "total_pnl": 0.0}
    result = await db.execute(
        select(
            func.count(Trade.id).label("total_trades"),
            func.avg(Trade.realized_pnl).label("avg_pnl"),
            func.sum(Trade.realized_pnl).label("total_pnl"),
        ).where(Trade.account_id.in_(account_ids))
    )
    row = result.one()
    return {
        "total_trades": row.total_trades or 0,
        "avg_pnl": float(row.avg_pnl or 0),
        "total_pnl": float(row.total_pnl or 0),
    }


@router.get("/slippage")
async def get_slippage_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Average slippage by execution algorithm — scoped to current user's orders."""
    from app.models.order import Order as OrderModel
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return []
    result = await db.execute(
        select(
            SlippageRecord.execution_algo,
            func.avg(SlippageRecord.slippage_bps).label("avg_bps"),
            func.count(SlippageRecord.id).label("count"),
        )
        .join(OrderModel, SlippageRecord.order_id == OrderModel.id)
        .where(OrderModel.account_id.in_(account_ids))
        .group_by(SlippageRecord.execution_algo)
    )
    rows = result.all()
    return [{"algo": r.execution_algo, "avg_bps": round(float(r.avg_bps or 0), 2), "count": r.count} for r in rows]


@router.get("/attribution")
async def get_pnl_attribution(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """P&L broken down by strategy — scoped to current user's accounts."""
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return []
    result = await db.execute(
        select(
            Trade.strategy_name,
            func.count(Trade.id).label("trades"),
            func.sum(Trade.realized_pnl).label("total_pnl"),
            func.avg(Trade.realized_pnl).label("avg_pnl"),
            func.sum(case((Trade.realized_pnl > 0, 1), else_=0)).label("wins"),
        )
        .where(Trade.account_id.in_(account_ids))
        .group_by(Trade.strategy_name)
        .order_by(func.sum(Trade.realized_pnl).desc())
    )
    rows = result.all()
    out = []
    for r in rows:
        total = float(r.total_pnl or 0)
        trades = r.trades or 0
        wins = r.wins or 0
        out.append({
            "strategy": r.strategy_name or "manual",
            "trades": trades,
            "total_pnl": round(total, 2),
            "avg_pnl": round(float(r.avg_pnl or 0), 2),
            "win_rate": round(wins / max(trades, 1), 3),
        })
    return out


@router.get("/macro")
async def get_macro_signals(current_user: User = Depends(get_current_user)):
    """Current macro environment signals from FRED (free, no API key)."""
    from app.ml.features.macro_signals import get_macro_snapshot_cached
    return await get_macro_snapshot_cached()


@router.get("/sentiment")
async def get_reddit_sentiment_endpoint(
    tickers: str | None = None,
    current_user: User = Depends(get_current_user),
):
    """Reddit WallStreetBets sentiment from Apewisdom (free, no key required)."""
    from app.ml.features.macro_signals import get_reddit_sentiment
    ticker_list = tickers.split(",") if tickers else None
    return await get_reddit_sentiment(ticker_list)


# ─── Correlation Matrix ───────────────────────────────────────────────────────

DEFAULT_SYMBOLS = ["SPY", "QQQ", "GLD", "TLT", "AAPL", "BTC/USD"]

ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"


async def _fetch_alpaca_bars(symbols: list[str], days: int) -> dict[str, list[float]]:
    """Fetch daily close prices from Alpaca for the given symbols.

    Returns a dict mapping symbol -> list of close prices (oldest first).
    Symbols that fail to fetch are omitted from the result.
    """
    start_dt = (datetime.now(timezone.utc) - timedelta(days=days + 10)).strftime("%Y-%m-%d")
    # Alpaca doesn't carry BTC/USD in the stock bars endpoint — filter to equity-like symbols
    equity_symbols = [s for s in symbols if "/" not in s]
    if not equity_symbols:
        return {}

    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    params = {
        "symbols": ",".join(equity_symbols),
        "timeframe": "1Day",
        "start": start_dt,
        "limit": days + 10,
        "feed": "iex",
        "adjustment": "raw",
    }

    prices: dict[str, list[float]] = {}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(ALPACA_DATA_URL, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            bars_map = data.get("bars", {})
            for sym, bars in bars_map.items():
                prices[sym] = [float(b["c"]) for b in bars]
    except Exception:
        pass
    return prices


@router.get("/correlation")
async def get_correlation_matrix(
    account_id: Optional[str] = Query(None),
    days: int = Query(30, ge=5, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Compute pairwise Pearson correlation matrix of daily returns
    for the user's current open positions.
    """
    # Gather symbols from user's positions
    symbols: list[str] = []
    try:
        acct_q = select(Account).where(Account.user_id == current_user.id, Account.is_active == True)
        if account_id:
            acct_q = acct_q.where(Account.id == account_id)
        acct_result = await db.execute(acct_q)
        accounts = acct_result.scalars().all()

        if accounts:
            account_ids = [a.id for a in accounts]
            pos_result = await db.execute(
                select(Position.symbol).where(Position.account_id.in_(account_ids)).distinct()
            )
            symbols = [row[0] for row in pos_result.all()]
    except Exception:
        symbols = []

    if not symbols:
        symbols = DEFAULT_SYMBOLS

    # Fetch price data from Alpaca (equity symbols only)
    prices_map = await _fetch_alpaca_bars(symbols, days)

    if not prices_map:
        return {
            "symbols": symbols,
            "matrix": [],
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "error": "Unable to fetch price data from Alpaca. Check API credentials.",
        }

    # Keep only symbols we have data for
    available_symbols = sorted(prices_map.keys())

    # Build DataFrame of close prices
    series_dict: dict[str, pd.Series] = {}
    for sym in available_symbols:
        closes = prices_map[sym]
        if len(closes) >= 5:
            series_dict[sym] = pd.Series(closes, dtype=float)

    if len(series_dict) < 2:
        return {
            "symbols": available_symbols,
            "matrix": [[1.0]] if len(series_dict) == 1 else [],
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    # Align series by index (use shortest length)
    min_len = min(len(s) for s in series_dict.values())
    df = pd.DataFrame({sym: s.iloc[-min_len:].values for sym, s in series_dict.items()})

    # Daily returns
    returns = df.pct_change().dropna()

    # Pearson correlation
    corr = returns.corr(method="pearson")
    final_symbols = list(corr.columns)
    matrix = [[round(float(corr.loc[r, c]), 4) for c in final_symbols] for r in final_symbols]

    return {
        "symbols": final_symbols,
        "matrix": matrix,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Tax Lots ─────────────────────────────────────────────────────────────────

ALPACA_QUOTES_URL = "https://data.alpaca.markets/v2/stocks/quotes/latest"


async def _fetch_latest_price(symbol: str) -> Optional[float]:
    """Try to get the latest ask price for a symbol from Alpaca."""
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                ALPACA_QUOTES_URL,
                headers=headers,
                params={"symbols": symbol, "feed": "iex"},
            )
            resp.raise_for_status()
            data = resp.json()
            quote = data.get("quotes", {}).get(symbol)
            if quote:
                # midpoint of bid/ask
                bid = float(quote.get("bp", 0))
                ask = float(quote.get("ap", 0))
                if ask > 0:
                    return (bid + ask) / 2.0
    except Exception:
        pass
    return None


@router.get("/tax-lots/{symbol}")
async def get_tax_lots(
    symbol: str,
    account_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Compute open tax lots for a symbol using FIFO matching of buys vs sells.
    Returns unrealized P&L, holding period, and HIFO/FIFO/LIFO recommendation.
    """
    # Resolve account IDs for this user
    acct_q = select(Account.id).where(Account.user_id == current_user.id, Account.is_active == True)
    if account_id:
        acct_q = acct_q.where(Account.id == account_id)
    acct_result = await db.execute(acct_q)
    account_ids = [row[0] for row in acct_result.all()]

    if not account_ids:
        raise HTTPException(status_code=404, detail="No accounts found for this user.")

    # Fetch all filled buy orders for this symbol
    buy_q = (
        select(Order)
        .where(
            Order.account_id.in_(account_ids),
            Order.symbol == symbol.upper(),
            Order.side == "buy",
            Order.status == "filled",
            Order.filled_qty > 0,
        )
        .order_by(Order.filled_at.asc())
    )
    buy_result = await db.execute(buy_q)
    buys = buy_result.scalars().all()

    # Fetch all filled sell orders
    sell_q = (
        select(Order)
        .where(
            Order.account_id.in_(account_ids),
            Order.symbol == symbol.upper(),
            Order.side == "sell",
            Order.status == "filled",
            Order.filled_qty > 0,
        )
        .order_by(Order.filled_at.asc())
    )
    sell_result = await db.execute(sell_q)
    sells = sell_result.scalars().all()

    if not buys:
        return {
            "symbol": symbol.upper(),
            "lots": [],
            "total_unrealized_pnl": 0.0,
            "recommended_method": "FIFO",
            "tax_savings_hifo_vs_fifo": 0.0,
        }

    # FIFO matching: consume sell quantity against earliest buys first
    # Build mutable lot list: each entry is [qty_remaining, cost_basis_per_share, filled_at, order_id]
    lots_raw = [
        {
            "qty": float(o.filled_qty),
            "cost": float(o.avg_fill_price) if o.avg_fill_price else 0.0,
            "acquired_at": o.filled_at,
            "order_id": o.id,
        }
        for o in buys
    ]

    # Total sell quantity to consume
    total_sold = sum(float(o.filled_qty) for o in sells)

    remaining_sell = total_sold
    for lot in lots_raw:
        if remaining_sell <= 0:
            break
        consumed = min(lot["qty"], remaining_sell)
        lot["qty"] -= consumed
        remaining_sell -= consumed

    # Open lots: those with remaining quantity > 0
    open_lots = [l for l in lots_raw if l["qty"] > 1e-9]

    if not open_lots:
        return {
            "symbol": symbol.upper(),
            "lots": [],
            "total_unrealized_pnl": 0.0,
            "recommended_method": "FIFO",
            "tax_savings_hifo_vs_fifo": 0.0,
        }

    # Fetch current price — fall back to latest fill price in buy orders
    current_price = await _fetch_latest_price(symbol.upper())
    if current_price is None and buys:
        last_fill = buys[-1].avg_fill_price
        current_price = float(last_fill) if last_fill else None

    now = datetime.now(timezone.utc)
    result_lots = []
    for i, lot in enumerate(open_lots):
        qty = float(lot["qty"])
        cost = float(lot["cost"])
        acquired = lot["acquired_at"]
        # Ensure timezone-aware
        if acquired and acquired.tzinfo is None:
            acquired = acquired.replace(tzinfo=timezone.utc)

        holding_days = (now - acquired).days if acquired else 0
        is_long_term = holding_days > 365

        if current_price is not None and cost > 0:
            unrealized_pnl = (current_price - cost) * qty
            unrealized_pct = ((current_price - cost) / cost) * 100.0
        else:
            unrealized_pnl = None
            unrealized_pct = None

        result_lots.append({
            "lot_id": lot["order_id"],
            "symbol": symbol.upper(),
            "quantity": round(qty, 8),
            "cost_basis": round(cost, 4),
            "acquired_date": acquired.isoformat() if acquired else None,
            "current_price": round(current_price, 4) if current_price is not None else None,
            "unrealized_pnl": round(unrealized_pnl, 2) if unrealized_pnl is not None else None,
            "unrealized_pct": round(unrealized_pct, 2) if unrealized_pct is not None else None,
            "holding_days": holding_days,
            "is_long_term": is_long_term,
        })

    total_unrealized_pnl = sum(
        l["unrealized_pnl"] for l in result_lots if l["unrealized_pnl"] is not None
    )

    # Recommend method based on gain/loss situation
    # HIFO (highest cost first) minimizes gains when selling — best when lots have gains
    # LIFO may be better in declining markets
    # Simple heuristic: if total P&L > 0 → HIFO saves the most taxes
    #                   if total P&L < 0 → FIFO (harvest losses early)
    if total_unrealized_pnl > 0:
        recommended_method = "HIFO"
    elif total_unrealized_pnl < 0:
        recommended_method = "FIFO"
    else:
        recommended_method = "FIFO"

    # Compute HIFO vs FIFO tax savings estimate (if we were to sell all open lots)
    # FIFO: sell lowest-cost lots first → highest gains
    # HIFO: sell highest-cost lots first → lowest gains
    if current_price is not None:
        fifo_lots_by_cost_asc = sorted(result_lots, key=lambda x: x["cost_basis"])
        hifo_lots_by_cost_desc = sorted(result_lots, key=lambda x: x["cost_basis"], reverse=True)

        fifo_gain = sum((current_price - l["cost_basis"]) * l["quantity"] for l in fifo_lots_by_cost_asc)
        hifo_gain = sum((current_price - l["cost_basis"]) * l["quantity"] for l in hifo_lots_by_cost_desc)
        # Tax savings = difference in taxable gain (assume ~20% cap gains rate for illustration)
        tax_savings_hifo_vs_fifo = round((fifo_gain - hifo_gain) * 0.20, 2)
    else:
        tax_savings_hifo_vs_fifo = 0.0

    return {
        "symbol": symbol.upper(),
        "lots": result_lots,
        "total_unrealized_pnl": round(float(total_unrealized_pnl), 2),
        "recommended_method": recommended_method,
        "tax_savings_hifo_vs_fifo": tax_savings_hifo_vs_fifo,
    }


# ─── Portfolio Greeks ─────────────────────────────────────────────────────────

_ALPACA_OPTIONS_BASE = "https://paper-api.alpaca.markets"

_OPTION_SYMBOL_RE = re.compile(
    r"^[A-Z]{1,6}\d{6}[CP]\d{8}$"
)


def _is_option_symbol(symbol: str) -> bool:
    """Return True if symbol looks like an OCC-formatted option symbol."""
    return bool(_OPTION_SYMBOL_RE.match(symbol.upper()))


async def _fetch_options_snapshots_for_symbols(symbols: list[str]) -> dict[str, dict]:
    """Fetch Alpaca options snapshots for a list of option symbols."""
    if not symbols:
        return {}
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        "accept": "application/json",
    }
    results: dict[str, dict] = {}
    BATCH = 50
    async with httpx.AsyncClient(timeout=20.0) as client:
        for i in range(0, len(symbols), BATCH):
            batch = symbols[i : i + BATCH]
            try:
                resp = await client.get(
                    f"{_ALPACA_OPTIONS_BASE}/v2/options/snapshots",
                    params={"symbols": ",".join(batch), "feed": "indicative"},
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results.update(data.get("snapshots") or {})
            except Exception:
                pass
    return results


async def _get_account_equity_for_user(
    account_id: Optional[str],
    current_user: "User",
    db: AsyncSession,
) -> float:
    """Sum equity across user accounts (or a specific account). Falls back to 0."""
    acct_q = select(Account).where(
        Account.user_id == current_user.id,
        Account.is_active == True,  # noqa: E712
    )
    if account_id:
        acct_q = acct_q.where(Account.id == account_id)
    acct_result = await db.execute(acct_q)
    accounts = acct_result.scalars().all()

    total_equity = 0.0
    for acct in accounts:
        if acct.broker == "alpaca" and acct.encrypted_key:
            try:
                from app.brokers.alpaca_orders import get_alpaca_account
                data = await get_alpaca_account(acct)
                total_equity += float(data.get("equity", 0))
            except Exception:
                pass
    return total_equity


@router.get("/portfolio-greeks")
async def get_portfolio_greeks(
    account_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Aggregate portfolio-level options Greeks across all open option positions.
    Returns net delta/gamma/theta/vega, targets, warnings, and per-position breakdown.
    """
    # Resolve accounts
    acct_q = select(Account).where(
        Account.user_id == current_user.id,
        Account.is_active == True,  # noqa: E712
    )
    if account_id:
        acct_q = acct_q.where(Account.id == account_id)
    acct_result = await db.execute(acct_q)
    accounts = acct_result.scalars().all()

    account_ids = [a.id for a in accounts]
    if not account_ids:
        raise HTTPException(status_code=404, detail="No active accounts found for this user.")

    # Fetch all open positions
    pos_result = await db.execute(
        select(Position).where(Position.account_id.in_(account_ids))
    )
    all_positions = pos_result.scalars().all()

    # Filter to option positions
    option_positions = [p for p in all_positions if _is_option_symbol(p.symbol)]

    # Fetch account equity
    account_equity = await _get_account_equity_for_user(account_id, current_user, db)

    if not option_positions:
        theta_target = account_equity * 0.0015
        delta_limit = 0.30 * account_equity / 100.0
        return {
            "net_delta": 0.0,
            "net_gamma": 0.0,
            "net_theta": 0.0,
            "net_vega": 0.0,
            "theta_target": round(theta_target, 2),
            "theta_pct_of_target": 0.0,
            "delta_limit": round(delta_limit, 2),
            "is_delta_neutral": True,
            "warnings": [],
            "position_count": 0,
            "options_positions": [],
            "account_equity": round(account_equity, 2),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    # Fetch snapshots for all option symbols
    opt_symbols = [p.symbol.upper() for p in option_positions]
    snapshots = await _fetch_options_snapshots_for_symbols(opt_symbols)

    # Aggregate Greeks
    net_delta = 0.0
    net_gamma = 0.0
    net_theta = 0.0
    net_vega = 0.0
    positions_out = []

    for pos in option_positions:
        sym = pos.symbol.upper()
        qty = float(pos.quantity)
        snap = snapshots.get(sym, {})
        greeks = snap.get("greeks") or {}
        iv = snap.get("impliedVolatility")

        delta = greeks.get("delta") or 0.0
        gamma = greeks.get("gamma") or 0.0
        theta = greeks.get("theta") or 0.0
        vega = greeks.get("vega") or 0.0

        # Multiply by quantity and 100 (contract multiplier)
        multiplier = qty * 100.0
        pos_delta = delta * multiplier
        pos_gamma = gamma * multiplier
        pos_theta = theta * multiplier
        pos_vega = vega * multiplier

        net_delta += pos_delta
        net_gamma += pos_gamma
        net_theta += pos_theta
        net_vega += pos_vega

        positions_out.append({
            "symbol": sym,
            "quantity": qty,
            "delta": round(delta, 4),
            "gamma": round(gamma, 4),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
            "iv": round(float(iv), 4) if iv is not None else None,
            "position_delta": round(pos_delta, 4),
            "position_gamma": round(pos_gamma, 4),
            "position_theta": round(pos_theta, 4),
            "position_vega": round(pos_vega, 4),
        })

    # Calculate targets
    theta_target = account_equity * 0.0015 if account_equity > 0 else 0.0
    delta_limit = 0.30 * account_equity / 100.0 if account_equity > 0 else 0.0
    theta_pct_of_target = (net_theta / theta_target * 100.0) if theta_target > 0 else 0.0
    is_delta_neutral = abs(net_delta) < delta_limit if delta_limit > 0 else True

    # Build warnings
    warnings = []
    if net_vega < -1000:
        warnings.append("Net vega exceeds -1000 — high IV risk")
    if not is_delta_neutral:
        warnings.append(f"Net delta ({net_delta:+.2f}) exceeds limit (±{delta_limit:.2f}) — portfolio not delta neutral")
    if theta_target > 0 and net_theta < theta_target * 0.5:
        warnings.append(f"Net theta (${net_theta:.2f}) is below 50% of target (${theta_target:.2f}) — consider adding premium")
    if account_equity == 0:
        warnings.append("Unable to fetch account equity — targets may be zero")

    return {
        "net_delta": round(net_delta, 4),
        "net_gamma": round(net_gamma, 4),
        "net_theta": round(net_theta, 4),
        "net_vega": round(net_vega, 4),
        "theta_target": round(theta_target, 2),
        "theta_pct_of_target": round(theta_pct_of_target, 2),
        "delta_limit": round(delta_limit, 2),
        "is_delta_neutral": is_delta_neutral,
        "warnings": warnings,
        "position_count": len(option_positions),
        "options_positions": positions_out,
        "account_equity": round(account_equity, 2),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/portfolio/snapshot")
async def get_portfolio_snapshot(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Aggregate portfolio KPIs for the dashboard snapshot widget — scoped to user."""
    account_ids = await _user_account_ids(db, current_user.id)

    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    if not account_ids:
        return {
            "total_pnl": 0.0, "today_pnl": 0.0, "today_pnl_trend": 0.0,
            "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "open_positions": 0,
        }

    # All-time realized PnL (scoped to user's accounts)
    total_pnl_result = await db.execute(
        select(func.coalesce(func.sum(Trade.realized_pnl), 0.0))
        .where(Trade.account_id.in_(account_ids))
    )
    total_pnl = float(total_pnl_result.scalar_one())

    # Today's realized PnL
    today_pnl_result = await db.execute(
        select(func.coalesce(func.sum(Trade.realized_pnl), 0.0)).where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc),
        )
    )
    today_pnl = float(today_pnl_result.scalar_one())

    # Yesterday's realized PnL (for trend)
    yesterday_pnl_result = await db.execute(
        select(func.coalesce(func.sum(Trade.realized_pnl), 0.0)).where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= datetime.combine(yesterday, datetime.min.time()).replace(tzinfo=timezone.utc),
            Trade.closed_at < datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc),
        )
    )
    yesterday_pnl = float(yesterday_pnl_result.scalar_one())

    # Win rate: positive trades / total trades (last 90 days)
    since_90 = datetime.now(timezone.utc) - timedelta(days=90)
    wins_result = await db.execute(
        select(
            func.count(Trade.id).label("total"),
            func.sum(case((Trade.realized_pnl > 0, 1), else_=0)).label("wins"),
        ).where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= since_90,
        )
    )
    wins_row = wins_result.one()
    total_trades = wins_row.total or 0
    wins = int(wins_row.wins or 0)
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    # Open positions count (scoped to user)
    open_pos_result = await db.execute(
        select(func.count(Position.id)).where(Position.account_id.in_(account_ids))
    )
    open_positions = int(open_pos_result.scalar_one() or 0)

    # Sharpe ratio: annualized from last 252 daily PnL values (scoped)
    daily_pnl_result = await db.execute(
        select(
            func.date_trunc("day", Trade.closed_at).label("day"),
            func.sum(Trade.realized_pnl).label("daily_pnl"),
        )
        .where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= datetime.now(timezone.utc) - timedelta(days=365),
        )
        .group_by(func.date_trunc("day", Trade.closed_at))
        .order_by(func.date_trunc("day", Trade.closed_at))
    )
    daily_rows = daily_pnl_result.all()
    sharpe = 0.0
    max_drawdown = 0.0
    if len(daily_rows) >= 5:
        daily_pnls = [float(r.daily_pnl) for r in daily_rows]
        s = pd.Series(daily_pnls)
        mean = s.mean()
        std = s.std()
        sharpe = (mean / std * (252 ** 0.5)) if std > 0 else 0.0
        # Max drawdown from cumulative PnL
        cum = s.cumsum()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max)
        max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    return {
        "total_pnl": round(total_pnl, 2),
        "today_pnl": round(today_pnl, 2),
        "today_pnl_trend": round(today_pnl - yesterday_pnl, 2),
        "sharpe": round(sharpe, 4),
        "win_rate": round(win_rate, 4),
        "max_drawdown": round(max_drawdown, 2),
        "open_positions": open_positions,
    }


@router.get("/equity-curve")
async def get_equity_curve(
    days: int = Query(365, ge=30, le=730, description="Lookback window in days"),
    initial_equity: float = Query(100_000.0, ge=1_000, description="Baseline equity to build curve from"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Daily cumulative equity curve built from realized trade P&L.

    Returns [{date, equity}] sorted ascending. The curve starts at
    initial_equity and adds each day's realized P&L going forward.
    Returns an empty list when there are no closed trades.
    """
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return []

    since = datetime.now(timezone.utc) - timedelta(days=days)

    result = await db.execute(
        select(
            func.date_trunc("day", Trade.closed_at).label("day"),
            func.sum(Trade.realized_pnl).label("daily_pnl"),
        )
        .where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= since,
            Trade.realized_pnl.isnot(None),
        )
        .group_by(func.date_trunc("day", Trade.closed_at))
        .order_by(func.date_trunc("day", Trade.closed_at))
    )
    rows = result.all()
    if not rows:
        return []

    equity = initial_equity
    curve = []
    for row in rows:
        equity += float(row.daily_pnl)
        day = row.day
        curve.append({
            "date": day.strftime("%Y-%m-%d") if hasattr(day, "strftime") else str(day)[:10],
            "equity": round(equity, 2),
        })
    return curve


@router.get("/monthly-returns")
async def get_monthly_returns(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Monthly realized P&L as a percentage of a $100k baseline.

    Returns [{month: "Jan 2024", ret: 2.3}] for the last 24 months,
    sorted oldest-first. Used to populate the monthly return heatmap.
    Returns an empty list when there are no closed trades.
    """
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return []

    since = datetime.now(timezone.utc) - timedelta(days=730)

    result = await db.execute(
        select(
            func.date_trunc("month", Trade.closed_at).label("month"),
            func.sum(Trade.realized_pnl).label("monthly_pnl"),
        )
        .where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= since,
            Trade.realized_pnl.isnot(None),
        )
        .group_by(func.date_trunc("month", Trade.closed_at))
        .order_by(func.date_trunc("month", Trade.closed_at))
    )
    rows = result.all()
    if not rows:
        return []

    # Build running equity to compute return % relative to start-of-month equity
    baseline = 100_000.0
    running_equity = baseline
    out = []
    for row in rows:
        monthly_pnl = float(row.monthly_pnl)
        ret_pct = round(monthly_pnl / max(running_equity, 1) * 100, 2)
        running_equity += monthly_pnl
        month = row.month
        out.append({
            "month": month.strftime("%b %Y") if hasattr(month, "strftime") else str(month)[:7],
            "ret": ret_pct,
        })
    return out


@router.get("/tearsheet")
async def get_tearsheet(
    days: int = Query(365, ge=90, le=730, description="Lookback window in days"),
    initial_equity: float = Query(100_000.0, ge=1_000, description="Baseline equity"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Fund-style tearsheet metrics for investor pitch.

    Computes full performance analytics from Trade records:
      - Sharpe, Sortino, Calmar, Omega ratio, Ulcer Index
      - Total return, annualised return, max drawdown
      - Win rate, profit factor, avg win/loss
      - Benchmark comparison vs SPY (via yfinance)
      - Monthly returns heatmap data
      - Equity curve and drawdown curve

    Returns 404 when no trade data exists (no mock data).
    """
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        raise HTTPException(status_code=404, detail="No accounts found")

    since = datetime.now(timezone.utc) - timedelta(days=days)

    # Fetch daily P&L
    result = await db.execute(
        select(
            func.date_trunc("day", Trade.closed_at).label("day"),
            func.sum(Trade.realized_pnl).label("daily_pnl"),
            func.count(Trade.id).label("n_trades"),
            func.sum(case((Trade.realized_pnl > 0, 1), else_=0)).label("n_wins"),
            func.avg(case((Trade.realized_pnl > 0, Trade.realized_pnl), else_=None)).label("avg_win"),
            func.avg(case((Trade.realized_pnl <= 0, Trade.realized_pnl), else_=None)).label("avg_loss"),
        )
        .where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= since,
            Trade.realized_pnl.isnot(None),
        )
        .group_by(func.date_trunc("day", Trade.closed_at))
        .order_by(func.date_trunc("day", Trade.closed_at))
    )
    rows = result.all()

    if not rows:
        raise HTTPException(status_code=404, detail="No trade data found in the requested period")

    # Build daily series
    daily_pnls = [float(r.daily_pnl) for r in rows]
    n_trades_total = sum(r.n_trades for r in rows)
    n_wins_total = sum(r.n_wins for r in rows)
    avg_win = float(next((r.avg_win for r in rows if r.avg_win is not None), 0) or 0)
    avg_loss = float(next((r.avg_loss for r in rows if r.avg_loss is not None), 0) or 0)

    s = pd.Series(daily_pnls)
    rf_daily = 0.05 / 252

    # Equity curve
    equity = initial_equity
    equity_curve = []
    drawdown_curve = []
    peak = initial_equity
    for i, (row, pnl) in enumerate(zip(rows, daily_pnls)):
        equity += pnl
        if equity > peak:
            peak = equity
        dd_pct = round((equity - peak) / peak * 100, 4) if peak > 0 else 0.0
        day = row.day
        day_str = day.strftime("%Y-%m-%d") if hasattr(day, "strftime") else str(day)[:10]
        equity_curve.append({"date": day_str, "equity": round(equity, 2)})
        drawdown_curve.append({"date": day_str, "drawdown_pct": dd_pct})

    # Performance metrics
    total_return = (equity - initial_equity) / initial_equity
    n_years = days / 365.0
    annualized_return = (1.0 + total_return) ** (1.0 / max(n_years, 0.01)) - 1.0
    max_dd = min(d["drawdown_pct"] for d in drawdown_curve) if drawdown_curve else 0.0

    daily_returns = s / initial_equity
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    sharpe = float((mean_ret - rf_daily) / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0

    downside_rets = daily_returns[daily_returns < rf_daily]
    downside_std = float(downside_rets.std()) if len(downside_rets) > 0 else 0.0
    sortino = float((mean_ret - rf_daily) / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0

    calmar = float(annualized_return / abs(max_dd / 100.0)) if max_dd < 0 else 0.0

    # Omega ratio
    gains = daily_returns[daily_returns > rf_daily].sum()
    losses = abs(daily_returns[daily_returns <= rf_daily].sum())
    omega = float(gains / losses) if losses > 0 else float(gains > 0) * 999.0

    # Ulcer index
    drawdowns_pct = pd.Series([d["drawdown_pct"] for d in drawdown_curve])
    ulcer = float(math.sqrt((drawdowns_pct ** 2).mean())) if len(drawdowns_pct) > 0 else 0.0

    win_rate = n_wins_total / max(n_trades_total, 1)
    profit_factor = abs(avg_win * n_wins_total / (avg_loss * max(n_trades_total - n_wins_total, 1))) if avg_loss != 0 else 0.0

    # Monthly returns
    monthly_result = await db.execute(
        select(
            func.date_trunc("month", Trade.closed_at).label("month"),
            func.sum(Trade.realized_pnl).label("monthly_pnl"),
        )
        .where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= since,
            Trade.realized_pnl.isnot(None),
        )
        .group_by(func.date_trunc("month", Trade.closed_at))
        .order_by(func.date_trunc("month", Trade.closed_at))
    )
    monthly_rows = monthly_result.all()
    running_eq = initial_equity
    monthly_returns = []
    for row in monthly_rows:
        mpnl = float(row.monthly_pnl)
        ret_pct = round(mpnl / max(running_eq, 1) * 100, 2)
        running_eq += mpnl
        month = row.month
        monthly_returns.append({
            "month": month.strftime("%b %Y") if hasattr(month, "strftime") else str(month)[:7],
            "ret": ret_pct,
        })

    # Benchmark SPY via yfinance (best-effort, non-blocking)
    benchmark_sharpe_spy = None
    benchmark_return_spy = None
    try:
        import yfinance as yf
        spy = yf.download("SPY", period=f"{days}d", interval="1d", auto_adjust=True, progress=False)
        if spy is not None and len(spy) > 10:
            spy_close = spy["Close"].squeeze()
            spy_rets = spy_close.pct_change().dropna()
            spy_total = float(spy_close.iloc[-1] / spy_close.iloc[0] - 1)
            spy_std = spy_rets.std()
            benchmark_sharpe_spy = round(
                float((spy_rets.mean() - rf_daily) / spy_std * math.sqrt(252)) if spy_std > 0 else 0.0, 4
            )
            benchmark_return_spy = round(spy_total * 100, 2)
    except Exception:
        pass

    return {
        "period_days": days,
        "n_trading_days": len(rows),
        "n_trades": n_trades_total,
        # Core metrics
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "omega_ratio": round(omega, 4),
        "ulcer_index": round(ulcer, 4),
        # Returns
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(annualized_return * 100, 2),
        "max_drawdown_pct": round(max_dd, 2),
        # Trade stats
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "avg_win_pct": round(avg_win / initial_equity * 100, 4),
        "avg_loss_pct": round(avg_loss / initial_equity * 100, 4),
        # Benchmark
        "benchmark_sharpe_spy": benchmark_sharpe_spy,
        "benchmark_return_spy": benchmark_return_spy,
        # Time series
        "monthly_returns": monthly_returns,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
