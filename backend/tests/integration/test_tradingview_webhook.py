"""TradingView webhook receiver (IMPROVEMENTS P2: webhook-IN only, no trading).

Security contract: disabled without a configured secret (503, never an open
sink), 401 on a wrong secret, and alerts are normalized + retrievable. The
receiver must never place orders.
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_disabled_without_secret(client, monkeypatch):
    monkeypatch.delenv("TRADINGVIEW_WEBHOOK_SECRET", raising=False)
    r = await client.post("/api/v1/webhooks/tradingview", json={"ticker": "SPY"})
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_rejects_wrong_secret(client, monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_WEBHOOK_SECRET", "tv-secret-1")
    r = await client.post("/api/v1/webhooks/tradingview",
                          json={"ticker": "SPY", "secret": "nope"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_rejects_non_object_body(client, monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_WEBHOOK_SECRET", "tv-secret-1")
    r = await client.post("/api/v1/webhooks/tradingview", json=["not", "an", "object"])
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_accepts_and_normalizes_alert(client, monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_WEBHOOK_SECRET", "tv-secret-1")
    r = await client.post("/api/v1/webhooks/tradingview", json={
        "secret": "tv-secret-1",
        "ticker": "nvda",
        "action": "BUY",
        "close": "181.25",
        "strategy": "supertrend-15m",
        "message": "ST flip long",
    })
    assert r.status_code == 200, r.text
    alert = r.json()["alert"]
    assert alert["symbol"] == "NVDA"
    assert alert["side"] == "buy"
    assert alert["price"] == pytest.approx(181.25)
    assert alert["strategy"] == "supertrend-15m"
    assert "secret" not in alert                      # never echo the secret

    r = await client.get("/api/v1/webhooks/tradingview/recent")
    assert r.status_code == 200
    alerts = r.json()["alerts"]
    assert alerts and alerts[0]["symbol"] == "NVDA"   # newest first
