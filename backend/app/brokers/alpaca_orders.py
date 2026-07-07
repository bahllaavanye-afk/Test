"""Submit, cancel and modify orders via Alpaca REST API."""
import logging
import time
from typing import Any, Dict, List, Optional

import httpx
from app.config import settings
from app.models.account import Account
from app.utils.security import decrypt_secret

ALPACA_PAPER = "https://paper-api.alpaca.markets"
ALPACA_LIVE = "https://api.alpaca.markets"

logger = logging.getLogger(__name__)


async def _headers(account: Account) -> Dict[str, str]:
    key = decrypt_secret(account.encrypted_key)
    secret = decrypt_secret(account.encrypted_secret)
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _base_url(account: Account) -> str:
    return ALPACA_LIVE if account.mode == "live" else ALPACA_PAPER


def _log_info(event: str, start_ts: float, **extra: Any) -> None:
    """Log structured INFO messages with execution duration."""
    duration_ms = int((time.time() - start_ts) * 1000)
    log_payload = {"event": event, "duration_ms": duration_ms, **extra}
    logger.info(event, extra=log_payload)


async def submit_alpaca_order(account: Account, order_data: Dict[str, Any]) -> Dict[str, Any]:
    """Submit an order to Alpaca. Returns Alpaca order response."""
    start_ts = time.time()
    headers = await _headers(account)
    base = _base_url(account)

    payload = {
        "symbol": order_data["symbol"],
        "qty": str(order_data.get("quantity")) if order_data.get("quantity") else None,
        "notional": str(order_data.get("notional")) if order_data.get("notional") else None,
        "side": order_data["side"],
        "type": order_data.get("order_type", "market"),
        "time_in_force": order_data.get("time_in_force", "gtc"),
        "limit_price": str(order_data["limit_price"]) if order_data.get("limit_price") else None,
        "stop_price": str(order_data["stop_price"]) if order_data.get("stop_price") else None,
    }

    # Add bracket legs if present
    if order_data.get("take_profit_price") or order_data.get("stop_loss_price"):
        payload["order_class"] = "bracket"
        if order_data.get("take_profit_price"):
            payload["take_profit"] = {"limit_price": str(order_data["take_profit_price"])}
        if order_data.get("stop_loss_price"):
            if order_data.get("trailing_stop_pct"):
                payload["stop_loss"] = {"trail_percent": str(order_data["trailing_stop_pct"])}
            else:
                payload["stop_loss"] = {"stop_price": str(order_data["stop_loss_price"])}
    elif order_data.get("trailing_stop_pct"):
        payload["type"] = "trailing_stop"
        payload["trail_percent"] = str(order_data["trailing_stop_pct"])

    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(f"{base}/v2/orders", json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()

    _log_info(
        "submit_alpaca_order",
        start_ts,
        signal_count=len(order_data),
        order_id=result.get("id"),
        pnl=None,  # P&L not available at order submission time
    )
    return result


async def cancel_alpaca_order(account: Account, broker_order_id: str) -> bool:
    """Cancel an existing Alpaca order."""
    start_ts = time.time()
    headers = await _headers(account)
    base = _base_url(account)

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.delete(f"{base}/v2/orders/{broker_order_id}", headers=headers)
        success = resp.status_code in (200, 204)

    _log_info(
        "cancel_alpaca_order",
        start_ts,
        order_id=broker_order_id,
        cancel_success=success,
        pnl=None,
    )
    return success


async def modify_alpaca_order(account: Account, broker_order_id: str, changes: Dict[str, Any]) -> Dict[str, Any]:
    """Modify an existing Alpaca order."""
    start_ts = time.time()
    headers = await _headers(account)
    base = _base_url(account)

    payload: Dict[str, str] = {}
    if changes.get("quantity"):
        payload["qty"] = str(changes["quantity"])
    if changes.get("limit_price"):
        payload["limit_price"] = str(changes["limit_price"])
    if changes.get("stop_price"):
        payload["stop_price"] = str(changes["stop_price"])

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.patch(f"{base}/v2/orders/{broker_order_id}", json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()

    _log_info(
        "modify_alpaca_order",
        start_ts,
        order_id=broker_order_id,
        changes_applied=len(payload),
        pnl=None,
    )
    return result


async def get_alpaca_positions(account: Account) -> List[Dict[str, Any]]:
    """Retrieve current Alpaca positions."""
    start_ts = time.time()
    headers = await _headers(account)
    base = _base_url(account)

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{base}/v2/positions", headers=headers)
        resp.raise_for_status()
        positions = resp.json()

    _log_info(
        "get_alpaca_positions",
        start_ts,
        position_count=len(positions),
        pnl=None,
    )
    return positions


async def get_alpaca_account(account: Account) -> Dict[str, Any]:
    """Retrieve Alpaca account details."""
    start_ts = time.time()
    headers = await _headers(account)
    base = _base_url(account)

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{base}/v2/account", headers=headers)
        resp.raise_for_status()
        account_info = resp.json()

    _log_info(
        "get_alpaca_account",
        start_ts,
        account_id=account_info.get("id"),
        pnl=None,
    )
    return account_info


# ── Options: multi-leg orders (paper-supported) ──────────────────────────────
# Alpaca supports options on the SAME keys/endpoints we already use — including
# multi-leg (order_class="mleg") on the paper venue. This is the "use Alpaca
# until TradeStation" path: real option legs, real (paper) fills, no new broker.


def build_occ_symbol(underlying: str, expiry, strike: float, option_type: str) -> str:
    """OCC option symbol: ROOT + YYMMDD + C/P + strike*1000 zero-padded to 8.

    e.g. SPY 2026-07-10 620 call → 'SPY260710C00620000'.
    """
    root = underlying.upper().strip()
    ymd = expiry.strftime("%y%m%d")
    cp = "C" if option_type.lower().startswith("c") else "P"
    return f"{root}{ymd}{cp}{int(round(strike * 1000)):08d}"


def pick_contract_by_delta(
    snapshots: Dict[str, Any], target_delta: float, option_type: str
) -> Optional[str]:
    """From an options-snapshots map {occ_symbol: snapshot}, pick the contract
    whose |greeks.delta| is closest to the target. Pure function (unit-tested).

    Puts have negative delta upstream; targets are specified as magnitudes
    (0.16Δ etc.) so compare absolute values. Contracts missing greeks are
    skipped — never guess a strike.
    """
    best_sym, best_err = None, 1e9
    for sym, snap in snapshots.items():
        greeks = (snap or {}).get("greeks") or {}
        d = greeks.get("delta")
        if d is None:
            continue
        err = abs(abs(float(d)) - abs(target_delta))
        if err < best_err:
            best_sym, best_err = sym, err
    return best_sym


async def _nearest_expiration(account: Account, underlying: str, dte: int) -> Optional[str]:
    """Closest listed expiration ≥ today+dte (same-day for 0DTE)."""
    from datetime import date, timedelta

    headers = await _headers(account)
    base = _base_url(account)
    target = date.today() + timedelta(days=dte)

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{base}/v2/assets/{underlying}/options",
            params={"expiration": target.isoformat()},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    if not data:
        return None
    # Assume the API returns a list of expiration dates sorted chronologically.
    return data[0].get("expiration")