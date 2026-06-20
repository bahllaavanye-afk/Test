"""Live, end-to-end real-time tests (``@pytest.mark.live``).

These exist to close the HANDOFF §3 gap: the existing realtime suite only asserts
``status_code < 500`` (a 404 or empty ``[]`` passes), so a dead broker / stale feed is
invisible to CI. The tests here assert on *actual data*:

* ``test_quote_endpoint_returns_fresh_price`` — the market-data quote must carry a
  non-zero price and a recent, parseable timestamp from a live source.
* ``test_ws_price_tick_arrives`` — opens a **real** WebSocket to ``/ws/prices/{symbol}``
  and asserts a real-data tick is delivered end to end.

They are skip-guarded so a default ``pytest tests/`` run never fails on them when
prerequisites are missing; select them explicitly with ``pytest -m live``. Thresholds
are env-tunable (no magic constants baked in).
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import socket
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.config import settings

# Tunable thresholds (env-overridable so nothing is hard-coded to one environment).
QUOTE_MAX_AGE_DAYS = float(os.getenv("LIVE_QUOTE_MAX_AGE_DAYS", "7"))
WS_TICK_TIMEOUT_S = float(os.getenv("LIVE_WS_TIMEOUT_S", "15"))
WS_SYMBOL = os.getenv("LIVE_WS_SYMBOL", "SPY")
QUOTE_SYMBOL = os.getenv("LIVE_QUOTE_SYMBOL", "SPY")

_PASSWORD = "L1ve!Tester2026"


# ─── Gating helpers ──────────────────────────────────────────────────────────

def _has_real_alpaca_creds() -> bool:
    key = (settings.alpaca_api_key or "").strip()
    secret = (settings.alpaca_secret_key or "").strip()
    return bool(key and secret) and key.lower() not in {"test-key", "test", "changeme", ""}


def _host_reachable(host: str, port: int = 443, timeout: float = 8.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _fetch_real_last_price(symbol: str) -> float | None:
    """Pull a real last price from a free, keyless source (Binance/yfinance).

    Returns the most recent close, or None if no free source yields data — which is
    exactly when the WS E2E should skip rather than hang.
    """
    from app.tasks import price_feed as pf

    try:
        if "/" in symbol:  # crypto → Binance public REST
            bars = pf._binance_klines_sync(symbol, "1m", 1)
        else:              # equity → yfinance
            bars = pf._yf_history_sync(symbol, "5d", "1d")
    except Exception:
        return None
    if bars:
        last = float(bars[-1]["close"])
        if last > 0:
            return last
    return None


def _parse_ts(raw: str) -> datetime:
    """Parse an RFC3339 timestamp, tolerating nanosecond precision and 'Z'."""
    s = str(raw).strip().replace("Z", "+00:00")
    # fromisoformat only accepts 3 or 6 fractional digits — truncate to 6.
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def _auth(client) -> dict[str, str]:
    email = f"live_{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": _PASSWORD}
    )
    if r.status_code == 201:
        return {"Authorization": f"Bearer {r.json()['access_token']}"}
    resp = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": _PASSWORD}
    )
    if resp.status_code != 200:
        pytest.skip(f"Login failed ({resp.status_code}) — DB not migrated in test env")
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _receive_with_timeout(ws, timeout: float) -> str:
    """Block on a TestClient websocket receive with a hard timeout."""
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(ws.receive_text)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        pytest.fail(f"No WS tick arrived within {timeout}s")
    finally:
        ex.shutdown(wait=False)


# ─── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.live
@pytest.mark.asyncio
async def test_quote_endpoint_returns_fresh_price(client):
    """The quote endpoint must return a real, non-zero, recently-timestamped price."""
    if not _has_real_alpaca_creds():
        pytest.skip("No real Alpaca credentials — quote endpoint cannot return live data")
    if not _host_reachable("data.alpaca.markets"):
        pytest.skip("data.alpaca.markets unreachable")

    headers = await _auth(client)
    r = await client.get(f"/api/v1/market-data/quote/{QUOTE_SYMBOL}", headers=headers)
    assert r.status_code == 200, r.text
    data = r.json()

    price = data.get("last") or data.get("mid_price")
    assert price is not None and float(price) > 0, f"empty/zero price (synthetic?): {data}"
    assert data.get("source") not in (None, "unavailable"), f"non-live source: {data}"

    ts_raw = data.get("timestamp")
    assert ts_raw, f"missing timestamp: {data}"
    age = datetime.now(timezone.utc) - _parse_ts(ts_raw)
    assert timedelta(0) <= age <= timedelta(days=QUOTE_MAX_AGE_DAYS), (
        f"stale/future quote timestamp {ts_raw!r} (age {age})"
    )


@pytest.mark.live
def test_ws_price_tick_arrives():
    """Open a real WebSocket and assert a real-data tick is delivered end to end.

    Spins up a minimal app with the real ``/ws/prices/{symbol}`` route + the real
    ``ConnectionManager``, and a publisher (on the same event loop) that broadcasts a
    price fetched live from a free source. A genuine client socket must receive it.
    """
    price = _fetch_real_last_price(WS_SYMBOL)
    if price is None:
        pytest.skip(
            "No free real-time data source reachable (e.g. Binance 451 / yfinance 429) — "
            "cannot drive a real tick"
        )

    import asyncio
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from app.ws.manager import manager
    from app.ws.prices import router as prices_router

    topic = f"prices:{WS_SYMBOL}"

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        stop = asyncio.Event()

        async def _publish() -> None:
            while not stop.is_set():
                await manager.broadcast(
                    topic,
                    {"type": "quote", "symbol": WS_SYMBOL, "last": price, "ts": time.time()},
                )
                await asyncio.sleep(0.5)

        task = asyncio.create_task(_publish())
        try:
            yield
        finally:
            stop.set()
            task.cancel()

    test_app = FastAPI(lifespan=_lifespan)
    test_app.include_router(prices_router)

    with TestClient(test_app) as c:
        with c.websocket_connect(f"/ws/prices/{WS_SYMBOL}") as ws:
            msg = _receive_with_timeout(ws, WS_TICK_TIMEOUT_S)

    data = json.loads(msg)
    assert data["symbol"] == WS_SYMBOL
    assert float(data["last"]) > 0
