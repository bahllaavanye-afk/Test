"""Inbound webhook receivers — TradingView alerts.

IMPROVEMENTS P2 (2026-06-29 review): TradingView has no public trade API, but
its alerts can POST here (charts → webhook-IN). This endpoint RECEIVES and
records alerts for visibility — it does NOT auto-trade them (paper-first;
alerts are unauthenticated third-party input and only ever advisory).

Security model: TradingView webhooks can't send custom headers, so the shared
secret rides in the JSON body ("secret"). With TRADINGVIEW_WEBHOOK_SECRET
unset the endpoint is disabled (503) — never an open unauthenticated sink.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.utils.logging import logger

# Constants
TRADINGVIEW_WEBHOOK_ENV = "TRADINGVIEW_WEBHOOK_SECRET"
ERROR_DISABLED = "TradingView webhook receiver disabled — set TRADINGVIEW_WEBHOOK_SECRET."
ERROR_MALFORMED_BODY = "Body must be a JSON object."
ERROR_BAD_SECRET = "Bad or missing webhook secret."
MAX_RECENT_ALERTS = 200
DEFAULT_RECENT_LIMIT = 50
MIN_RECENT_LIMIT = 1
REDIS_ALERT_CHANNEL = "tradingview:alerts"

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Ring buffer of the most recent alerts (process-local; visibility, not storage
# of record). A dead Redis must not break the receiver.
_RECENT_ALERTS: list[dict] = []


def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
    """Best-effort normalization of TradingView's free-form alert JSON."""
    return {
        "symbol": str(payload.get("ticker") or payload.get("symbol") or "").upper() or None,
        "side": (str(payload.get("action") or payload.get("side") or "").lower() or None),
        "price": _float_or_none(payload.get("price") or payload.get("close")),
        "strategy": payload.get("strategy") or payload.get("indicator"),
        "message": str(payload.get("message") or payload.get("comment") or "")[:500] or None,
        "received_at": datetime.now(timezone.utc).isoformat(),
    }


def _float_or_none(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


@router.post("/tradingview")
async def receive_tradingview_alert(request: Request) -> dict:
    secret = os.environ.get(TRADINGVIEW_WEBHOOK_ENV, "").strip()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ERROR_DISABLED,
        )

    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
    except Exception:  # noqa: BLE001 — malformed body is a client error
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=ERROR_MALFORMED_BODY,
        )

    if str(payload.get("secret") or "") != secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_BAD_SECRET,
        )

    alert = _normalize(payload)
    _RECENT_ALERTS.append(alert)
    del _RECENT_ALERTS[:-MAX_RECENT_ALERTS]
    logger.info(
        "tradingview alert received",
        symbol=alert["symbol"],
        side=alert["side"],
        strategy=str(alert["strategy"])[:40],
    )

    # Best-effort fan-out to Redis subscribers (strategies/dashboards may listen).
    try:
        from app.redis_client import get_redis

        r = get_redis()
        if r is not None:
            import json as _json

            await r.publish(REDIS_ALERT_CHANNEL, _json.dumps(alert))
    except Exception as exc:  # noqa: BLE001 — receiver must not depend on Redis
        logger.debug("tradingview alert: redis publish skipped", error=str(exc))

    return {"ok": True, "alert": alert}


@router.get("/tradingview/recent")
async def recent_tradingview_alerts(limit: int = DEFAULT_RECENT_LIMIT) -> dict:
    """Most recent received alerts (process-local ring buffer)."""
    limit = max(MIN_RECENT_LIMIT, min(limit, MAX_RECENT_ALERTS))
    return {"alerts": _RECENT_ALERTS[-limit:][::-1], "count": len(_RECENT_ALERTS)}