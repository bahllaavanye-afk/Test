"""Submit, cancel and modify orders via Alpaca REST API."""
import logging
import time
from typing import Any, Dict, List

import httpx
from app.config import settings
from app.models.account import Account
from app.utils.security import decrypt_secret

ALPACA_PAPER = "https://paper-api.alpaca.markets"
ALPACA_LIVE = "https://api.alpaca.markets"

logger = logging.getLogger(__name__)

# Simple in‑memory caches
_headers_cache: Dict[int, Dict[str, str]] = {}
_client_cache: Dict[str, httpx.AsyncClient] = {}


async def _headers(account: Account) -> Dict[str, str]:
    """Return authentication headers, cached per account."""
    cached = _headers_cache.get(account.id)
    if cached:
        return cached
    key = decrypt_secret(account.encrypted_key)
    secret = decrypt_secret(account.encrypted_secret)
    hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    _headers_cache[account.id] = hdr
    return hdr


def _base_url(account: Account) -> str:
    """Base URL depending on live or paper mode."""
    return ALPACA_LIVE if account.mode == "live" else ALPACA_PAPER


def _get_client(base: str, timeout: int = 15) -> httpx.AsyncClient:
    """Retrieve a reusable AsyncClient for the given base URL."""
    client = _client_cache.get(base)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=timeout)
        _client_cache[base] = client
    return client


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

    client = _get_client(base, timeout=15)
    resp = await client.post(f"{base}/v2/orders", json=payload, headers=headers)
    resp.raise_for_status()
    result = resp.json()

    duration_ms = int((time.time() - start_ts) * 1000)
    logger.info(
        "submit_alpaca_order",
        extra={
            "signal_count": len(order_data),
            "duration_ms": duration_ms,
            "order_id": result.get("id"),
            "pnl": None,
        },
    )
    return result


async def cancel_alpaca_order(account: Account, broker_order_id: str) -> bool:
    """Cancel an existing Alpaca order."""
    start_ts = time.time()
    headers = await _headers(account)
    base = _base_url(account)

    client = _get_client(base, timeout=10)
    resp = await client.delete(f"{base}/v2/orders/{broker_order_id}", headers=headers)
    success = resp.status_code in (200, 204)

    duration_ms = int((time.time() - start_ts) * 1000)
    logger.info(
        "cancel_alpaca_order",
        extra={
            "order_id": broker_order_id,
            "duration_ms": duration_ms,
            "cancel_success": success,
            "pnl": None,
        },
    )
    return success


async def modify_alpaca_order(account: Account, broker_order_id: str, changes: Dict[str, Any]) -> Dict[str, Any]:
    """Modify an existing Alpaca order."""
    start_ts = time.time()
    headers = await _headers(account)
    base = _base_url(account)

    payload = {}
    if changes.get("quantity"):
        payload["qty"] = str(changes["quantity"])
    if changes.get("limit_price"):
        payload["limit_price"] = str(changes["limit_price"])
    if changes.get("stop_price"):
        payload["stop_price"] = str(changes["stop_price"])

    client = _get_client(base, timeout=10)
    resp = await client.patch(f"{base}/v2/orders/{broker_order_id}", json=payload, headers=headers)
    resp.raise_for_status()
    result = resp.json()

    duration_ms = int((time.time() - start_ts) * 1000)
    logger.info(
        "modify_alpaca_order",
        extra={
            "order_id": broker_order_id,
            "duration_ms": duration_ms,
            "changes_applied": len(payload),
            "pnl": None,
        },
    )
    return result


async def get_alpaca_positions(account: Account) -> List[Dict[str, Any]]:
    """Retrieve current Alpaca positions."""
    start_ts = time.time()
    headers = await _headers(account)
    base = _base_url(account)

    client = _get_client(base, timeout=10)
    resp = await client.get(f"{base}/v2/positions", headers=headers)
    resp.raise_for_status()
    positions = resp.json()

    duration_ms = int((time.time() - start_ts) * 1000)
    logger.info(
        "get_alpaca_positions",
        extra={
            "position_count": len(positions),
            "duration_ms": duration_ms,
            "pnl": None,
        },
    )
    return positions


async def get_alpaca_account(account: Account) -> Dict[str, Any]:
    """Retrieve Alpaca account details."""
    start_ts = time.time()
    headers = await _headers(account)
    base = _base_url(account)

    client = _get_client(base, timeout=10)
    resp = await client.get(f"{base}/v2/account", headers=headers)
    resp.raise_for_status()
    account_info = resp.json()

    duration_ms = int((time.time() - start_ts) * 1000)
    logger.info(
        "get_alpaca_account",
        extra={
            "account_id": account_info.get("id"),
            "duration_ms": duration_ms,
            "pnl": None,
        },
    )
    return account_info