"""Submit, cancel and modify orders via Alpaca REST API."""
import logging
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Union

import httpx
from app.config import settings
from app.models.account import Account
from app.utils.security import decrypt_secret

ALPACA_PAPER = "https://paper-api.alpaca.markets"
ALPACA_LIVE = "https://api.alpaca.markets"

logger = logging.getLogger(__name__)


async def _headers(account: Account) -> Dict[str, str]:
    """
    Build the authentication headers required by Alpaca.

    Args:
        account: The account object containing encrypted API credentials.

    Returns:
        A dictionary with the ``APCA-API-KEY-ID`` and ``APCA-API-SECRET-KEY`` headers.
    """
    key = decrypt_secret(account.encrypted_key)
    secret = decrypt_secret(account.encrypted_secret)
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _base_url(account: Account) -> str:
    """
    Choose the appropriate Alpaca base URL based on the account mode.

    Args:
        account: The account object whose mode determines the endpoint.

    Returns:
        The live URL if ``account.mode`` is ``"live"``, otherwise the paper URL.
    """
    return ALPACA_LIVE if account.mode == "live" else ALPACA_PAPER


async def submit_alpaca_order(account: Account, order_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit an order to Alpaca.

    Args:
        account: Account containing authentication details and mode.
        order_data: Dictionary describing the order; keys include ``symbol``, ``quantity``,
            ``notional``, ``side``, ``order_type``, ``time_in_force``, ``limit_price``,
            ``stop_price``, and optional bracket parameters.

    Returns:
        The JSON response from Alpaca representing the created order.
    """
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

    duration_ms = int((time.time() - start_ts) * 1000)
    logger.info(
        "submit_alpaca_order",
        extra={
            "signal_count": len(order_data),
            "duration_ms": duration_ms,
            "order_id": result.get("id"),
            "pnl": None,  # P&L not available at order submission time
        },
    )
    return result


async def cancel_alpaca_order(account: Account, broker_order_id: str) -> bool:
    """
    Cancel an existing Alpaca order.

    Args:
        account: Account containing authentication details.
        broker_order_id: The Alpaca order identifier to cancel.

    Returns:
        ``True`` if the cancellation succeeded (HTTP 200/204), otherwise ``False``.
    """
    start_ts = time.time()
    headers = await _headers(account)
    base = _base_url(account)

    async with httpx.AsyncClient(timeout=10) as client:
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
    """
    Modify an existing Alpaca order.

    Args:
        account: Account containing authentication details.
        broker_order_id: The identifier of the order to modify.
        changes: Dictionary of fields to change; supported keys are ``quantity``,
            ``limit_price`` and ``stop_price``.

    Returns:
        The JSON response from Alpaca representing the updated order.
    """
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
    """
    Retrieve current Alpaca positions.

    Args:
        account: Account containing authentication details.

    Returns:
        A list of position dictionaries as returned by Alpaca.
    """
    start_ts = time.time()
    headers = await _headers(account)
    base = _base_url(account)

    async with httpx.AsyncClient(timeout=10) as client:
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
    """
    Retrieve Alpaca account details.

    Args:
        account: Account containing authentication details.

    Returns:
        A dictionary with account information as provided by Alpaca.
    """
    start_ts = time.time()
    headers = await _headers(account)
    base = _base_url(account)

    async with httpx.AsyncClient(timeout=10) as client:
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


# ── Options: multi-leg orders (paper-supported) ──────────────────────────────
# Alpaca supports options on the SAME keys/endpoints we already use — including
# multi-leg (order_class="mleg") on the paper venue. This is the "use Alpaca
# until TradeStation" path: real option legs, real (paper) fills, no new broker.


def build_occ_symbol(underlying: str, expiry: date, strike: float, option_type: str) -> str:
    """
    Construct an OCC option symbol.

    The format is ``ROOT`` + ``YYMMDD`` + ``C``/``P`` + ``strike*1000`` zero‑padded
    to eight digits.

    Example:
        SPY 2026‑07‑10 620 call → ``'SPY260710C00620000'``.

    Args:
        underlying: Underlying ticker symbol (e.g., ``"SPY"``).
        expiry: Expiration date.
        strike: Strike price.
        option_type: ``"call"`` or ``"put"`` (case‑insensitive).

    Returns:
        The OCC formatted option symbol.
    """
    root = underlying.upper().strip()
    ymd = expiry.strftime("%y%m%d")
    cp = "C" if option_type.lower().startswith("c") else "P"
    return f"{root}{ymd}{cp}{int(round(strike * 1000)):08d}"


def pick_contract_by_delta(
    snapshots: Dict[str, Any],
    target_delta: float,
    option_type: str,
) -> Optional[str]:
    """
    Choose the option contract whose delta is closest to a target value.

    The function operates on a mapping of OCC symbols to snapshot dictionaries
    that contain a ``greeks`` sub‑dictionary. Contracts lacking a delta are
    ignored.

    Args:
        snapshots: Mapping of OCC symbol → snapshot data.
        target_delta: Desired delta magnitude (e.g., ``0.16`` for a 16 Δ contract).
        option_type: ``"call"`` or ``"put"``; currently unused but retained for API
            compatibility.

    Returns:
        The OCC symbol of the best matching contract, or ``None`` if no suitable
        contract is found.
    """
    best_sym: Optional[str] = None
    best_err = 1e9
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
    """
    Find the closest listed expiration date that is at least ``dte`` days away.

    Args:
        account: Account containing authentication details.
        underlying: Underlying ticker symbol.
        dte: Desired days‑to‑expiration; ``0`` means same‑day expiration.

    Returns:
        The expiration date as an ISO‑8601 string (e.g., ``'2023-12-15'``) or
        ``None`` if no suitable expiration is found.
    """
    headers = await _headers(account)
    base = _base_url(account)
    target = date.today() + timedelta(days=dte)

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{base}/v2/assets/{underlying}/options", headers=headers)
        resp.raise_for_status()
        data = resp.json()

    expirations = sorted(set(item["expiration_date"] for item in data if "expiration_date" in item))
    for exp in expirations:
        exp_date = date.fromisoformat(exp)
        if exp_date >= target:
            return exp
    return None