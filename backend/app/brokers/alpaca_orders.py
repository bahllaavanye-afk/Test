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


async def _headers(account: Account) -> Dict[str, str]:
    key = decrypt_secret(account.encrypted_key)
    secret = decrypt_secret(account.encrypted_secret)
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _base_url(account: Account) -> str:
    return ALPACA_LIVE if account.mode == "live" else ALPACA_PAPER


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
    """Cancel an existing Alpaca order."""
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
    """Retrieve current Alpaca positions."""
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
    """Retrieve Alpaca account details."""
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

def build_occ_symbol(underlying: str, expiry, strike: float, option_type: str) -> str:
    """OCC option symbol: ROOT + YYMMDD + C/P + strike*1000 zero-padded to 8.

    e.g. SPY 2026-07-10 620 call → 'SPY260710C00620000'.
    """
    root = underlying.upper().strip()
    ymd = expiry.strftime("%y%m%d")
    cp = "C" if option_type.lower().startswith("c") else "P"
    return f"{root}{ymd}{cp}{int(round(strike * 1000)):08d}"


def pick_contract_by_delta(snapshots: Dict[str, Any], target_delta: float,
                           option_type: str) -> str | None:
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


async def _nearest_expiration(account: Account, underlying: str, dte: int) -> str | None:
    """Closest listed expiration ≥ today+dte (same-day for 0DTE)."""
    from datetime import date, timedelta

    headers = await _headers(account)
    base = _base_url(account)
    target = date.today() + timedelta(days=dte)
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{base}/v2/options/contracts",
            headers=headers,
            params={
                "underlying_symbols": underlying,
                "expiration_date_gte": target.isoformat(),
                "limit": 100,
                "status": "active",
            },
        )
        resp.raise_for_status()
        contracts = resp.json().get("option_contracts") or []
    expiries = sorted({c["expiration_date"] for c in contracts})
    return expiries[0] if expiries else None


async def resolve_leg_symbol(account: Account, underlying: str, leg: Dict[str, Any]) -> str | None:
    """Resolve one leg spec (delta or explicit strike + dte) to an OCC symbol.

    Explicit strike → build the symbol directly against the nearest expiry.
    Delta target → pull the expiry's contracts, then their snapshots (indicative
    feed carries greeks on paper), and pick the closest |delta|. Returns None
    when anything is missing — the caller degrades to the alert path.
    """
    from datetime import date

    expiry = await _nearest_expiration(account, underlying, int(leg.get("dte", 30)))
    if not expiry:
        return None
    expiry_date = date.fromisoformat(expiry)

    if leg.get("strike"):
        return build_occ_symbol(underlying, expiry_date, float(leg["strike"]), leg["option_type"])

    headers = await _headers(account)
    base = _base_url(account)
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{base}/v2/options/contracts",
            headers=headers,
            params={
                "underlying_symbols": underlying,
                "expiration_date": expiry,
                "type": leg["option_type"],
                "limit": 300,
                "status": "active",
            },
        )
        resp.raise_for_status()
        contracts = resp.json().get("option_contracts") or []
        if not contracts:
            return None
        symbols = ",".join(c["symbol"] for c in contracts[:100])
        snap_resp = await client.get(
            "https://data.alpaca.markets/v1beta1/options/snapshots",
            headers=headers,
            params={"symbols": symbols, "feed": "indicative"},
        )
        snapshots = (snap_resp.json() or {}).get("snapshots", {}) if snap_resp.status_code == 200 else {}
    return pick_contract_by_delta(snapshots, float(leg.get("delta") or 0.5), leg["option_type"])


async def submit_alpaca_multileg_order(
    account: Account, underlying: str, legs: List[Dict[str, Any]], quantity: int = 1
) -> Dict[str, Any] | None:
    """Submit a multi-leg options order (order_class='mleg') on Alpaca.

    Works on the paper venue with our existing keys. Legs: dicts with side,
    option_type, dte, and delta or strike; ratio defaults to 1. Returns the
    order response, or None when any leg can't be resolved (caller alerts).
    """
    resolved = []
    for leg in legs:
        sym = await resolve_leg_symbol(account, underlying, leg)
        if not sym:
            # stdlib logger — kwargs like structlog's raise TypeError and would
            # crash the caller exactly when the order needs handling (found by
            # test_multileg_broker_rejection_returns_none).
            logger.info("mleg leg unresolved — degrading to alert (%s %s)",
                        underlying, str(leg)[:80])
            return None
        resolved.append({
            "symbol": sym,
            "ratio_qty": str(int(leg.get("ratio", 1))),
            "side": leg["side"],
            "position_intent": "buy_to_open" if leg["side"] == "buy" else "sell_to_open",
        })

    headers = await _headers(account)
    base = _base_url(account)
    payload = {
        "order_class": "mleg",
        "qty": str(quantity),
        "type": "market",
        "time_in_force": "day",
        "legs": resolved,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(f"{base}/v2/orders", json=payload, headers=headers)
        if resp.status_code not in (200, 201):
            logger.warning("mleg order rejected: %s %s", resp.status_code, resp.text[:200])
            return None
        result = resp.json()
    logger.info("Alpaca multi-leg options order submitted: id=%s legs=%d underlying=%s",
                result.get("id"), len(resolved), underlying)
    return result
