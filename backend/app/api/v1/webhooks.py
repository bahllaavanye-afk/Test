"""Inbound webhook receivers — TradingView alerts.

IMPROVEMENTS P2 (2026-06-29 review): TradingView has no public trade API, but
its alerts can POST here (charts → webhook-IN). This endpoint RECEIVES and
records alerts for visibility — it does NOT auto-trade them (paper-first;
alerts are unauthenticated third‑party input and only ever advisory).

Security model: TradingView webhooks can't send custom headers, so the shared
secret rides in the JSON body ("secret"). With TRADINGVIEW_WEBHOOK_SECRET
unset the endpoint is disabled (503) — never an open unauthenticated sink.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.utils.logging import logger

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Ring buffer of the most recent alerts (process‑local; visibility, not storage
# of record). A dead Redis must not break the receiver.
_RECENT_ALERTS: list[dict] = []
_MAX_RECENT = 200

# Global counters for monitoring
_TOTAL_ALERTS: int = 0


def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
    """Best‑effort normalization of TradingView's free‑form alert JSON."""
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
    start_time = time.perf_counter()

    secret = os.getenv("TRADINGVIEW_WEBHOOK_SECRET", "").strip()
    if not secret:
        logger.error(
            "TradingView webhook receiver disabled",
            reason="missing_secret",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TradingView webhook receiver disabled — set TRADINGVIEW_WEBHOOK_SECRET.",
        )

    # Parse JSON payload
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to decode TradingView webhook JSON",
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Body must be a valid JSON object.",
        )
    except Exception as exc:  # Unexpected parsing error
        logger.error(
            "Unexpected error reading TradingView webhook body",
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error processing request body.",
        )

    if not isinstance(payload, dict):
        logger.error(
            "TradingView webhook payload is not a JSON object",
            payload_type=type(payload).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Body must be a JSON object.",
        )

    # Verify secret
    if str(payload.get("secret") or "") != secret:
        logger.warning(
            "Invalid TradingView webhook secret",
            provided_secret=str(payload.get("secret")),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bad or missing webhook secret.",
        )

    # Normalize and store alert
    alert = _normalize(payload)
    _RECENT_ALERTS.append(alert)
    del _RECENT_ALERTS[:-_MAX_RECENT]

    # Update monitoring counters
    global _TOTAL_ALERTS
    _TOTAL_ALERTS += 1

    # Structured logging of the received alert with monitoring metrics
    exec_time = time.perf_counter() - start_time
    logger.info(
        "tradingview alert received",
        symbol=alert["symbol"],
        side=alert["side"],
        strategy=str(alert["strategy"])[:40],
        signal_count=_TOTAL_ALERTS,
        exec_time_ms=round(exec_time * 1000, 2),
        pnl=alert.get("pnl"),
    )

    # Fan‑out to Redis subscribers (strategies/dashboards may listen)
    try:
        from app.redis_client import get_redis

        r = get_redis()
        if r is not None:
            await r.publish("tradingview:alerts", json.dumps(alert))
    except Exception as exc:  # Redis is optional; failures are logged only
        logger.debug(
            "tradingview alert: redis publish skipped",
            error=str(exc),
        )

    return {"ok": True, "alert": alert}


@router.get("/tradingview/recent")
async def recent_tradingview_alerts(limit: int = 50) -> dict:
    """Most recent received alerts (process‑local ring buffer)."""
    limit = max(1, min(limit, _MAX_RECENT))
    return {"alerts": _RECENT_ALERTS[-limit:][::-1], "count": len(_RECENT_ALERTS)}