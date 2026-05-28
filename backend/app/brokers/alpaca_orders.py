"""Submit, cancel and modify orders via Alpaca REST API."""
import httpx
from app.config import settings
from app.utils.security import decrypt_secret
from app.models.account import Account

ALPACA_PAPER = "https://paper-api.alpaca.markets"
ALPACA_LIVE = "https://api.alpaca.markets"


async def _headers(account: Account) -> dict:
    key = decrypt_secret(account.encrypted_key)
    secret = decrypt_secret(account.encrypted_secret)
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _base_url(account: Account) -> str:
    return ALPACA_LIVE if account.mode == "live" else ALPACA_PAPER


async def submit_alpaca_order(account: Account, order_data: dict) -> dict:
    """Submit an order to Alpaca. Returns Alpaca order response."""
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
        return resp.json()


async def cancel_alpaca_order(account: Account, broker_order_id: str) -> bool:
    headers = await _headers(account)
    base = _base_url(account)
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.delete(f"{base}/v2/orders/{broker_order_id}", headers=headers)
        return resp.status_code in (200, 204)


async def modify_alpaca_order(account: Account, broker_order_id: str, changes: dict) -> dict:
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
        return resp.json()


async def get_alpaca_positions(account: Account) -> list[dict]:
    headers = await _headers(account)
    base = _base_url(account)
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{base}/v2/positions", headers=headers)
        resp.raise_for_status()
        return resp.json()


async def get_alpaca_account(account: Account) -> dict:
    headers = await _headers(account)
    base = _base_url(account)
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{base}/v2/account", headers=headers)
        resp.raise_for_status()
        return resp.json()
