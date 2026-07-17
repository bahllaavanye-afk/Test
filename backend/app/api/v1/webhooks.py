"""Inbound webhook receivers — TradingView alerts.

IMPROVEMENTS P2 (2026-06-29 review): TradingView has no public trade API, but
its alerts can POST here (charts → webhook‑IN). This endpoint RECEIVES and
records alerts for visibility — it does NOT auto‑trade them (paper‑first;
alerts are unauthenticated third‑party input and only ever advisory).

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

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Ring buffer of the most recent alerts (process‑local; visibility, not storage
# of record). A dead Redis must not break the receiver.
_RECENT_ALERTS: list[dict] = []
_MAX_RECENT = 200


def _normalize(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Best‑effort normalization of TradingView's free‑form alert JSON.

    Handles ``None`` payloads by treating them as empty dictionaries.
    """
    if payload is None:
        payload = {}
    return {
        "symbol": (
            str(payload.get("ticker") or payload.get("symbol") or "").upper()
            or None
        ),
        "side": (
            str(payload.get("action") or payload.get("side") or "").lower()
            or None
        ),
        "price": _float_or_none(payload.get("price") or payload.get("close")),
        "strategy": payload.get("strategy") or payload.get("indicator"),
        "message": (
            str(payload.get("message") or payload.get("comment") or "")[:500]
            or None
        ),
        "received_at": datetime.now(timezone.utc).isoformat(),
    }


def _float_or_none(v: Any) -> float | None:
    """Convert *v* to ``float`` when possible, otherwise return ``None``."""
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


@router.post("/tradingview")
async def receive_tradingview_alert(request: Request) -> dict:
    """Receive a TradingView webhook, validate it, store a normalized copy,
    and optionally fan‑out to Redis. Returns a simple acknowledgement payload.
    """
    secret = os.environ.get("TRADINGVIEW_WEBHOOK_SECRET", "").strip()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TradingView webhook receiver disabled — set TRADINGVIEW_WEBHOOK_SECRET.",
        )

    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
    except Exception:  # noqa: BLE001 — malformed body is a client error
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Body must be a JSON object.",
        )

    if str(payload.get("secret") or "") != secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bad or missing webhook secret.",
        )

    alert = _normalize(payload)

    # Maintain ring buffer size safely
    _RECENT_ALERTS.append(alert)
    if len(_RECENT_ALERTS) > _MAX_RECENT:
        # Keep only the newest _MAX_RECENT entries
        del _RECENT_ALERTS[:-_MAX_RECENT]

    logger.info(
        "tradingview alert received",
        symbol=alert["symbol"],
        side=alert["side"],
        strategy=str(alert["strategy"])[:40],
    )

    # Best‑effort fan‑out to Redis subscribers (strategies/dashboards may listen).
    try:
        from app.redis_client import get_redis

        r = get_redis()
        if r is not None:
            import json as _json

            await r.publish("tradingview:alerts", _json.dumps(alert))
    except Exception as exc:  # noqa: BLE001 — receiver must not depend on Redis
        logger.debug("tradingview alert: redis publish skipped", error=str(exc))

    return {"ok": True, "alert": alert}


@router.get("/tradingview/recent")
async def recent_tradingview_alerts(limit: int | None = 50) -> dict:
    """Most recent received alerts (process‑local ring buffer).

    *limit* is clamped to the range ``[1, _MAX_RECENT]``. ``None`` defaults to
    ``50``. Returns alerts in reverse chronological order.
    """
    if limit is None:
        limit = 50
    # Ensure *limit* is an integer; FastAPI already validates, but guard against
    # accidental misuse in code.
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50

    limit = max(1, min(limit, _MAX_RECENT))
    alerts_slice = _RECENT_ALERTS[-limit:] if _RECENT_ALERTS else []
    return {"alerts": alerts_slice[::-1], "count": len(_RECENT_ALERTS)}