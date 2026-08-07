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
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.utils.logging import logger

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Ring buffer of the most recent alerts (process-local; visibility, not storage
# of record). A dead Redis must not break the receiver.
_RECENT_ALERTS: list[dict] = []
_MAX_RECENT = 200

# Global counters for monitoring
_TOTAL_ALERTS: int = 0


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


def _get_secret() -> str:
    """Retrieve the webhook secret from the environment."""
    return os.environ.get("TRADINGVIEW_WEBHOOK_SECRET", "").strip()


def _ensure_secret_enabled(secret: str) -> None:
    """Raise HTTPException if the webhook receiver is disabled."""
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TradingView webhook receiver disabled — set TRADINGVIEW_WEBHOOK_SECRET.",
        )


async def _parse_payload(request: Request) -> dict:
    """Parse JSON body and ensure it is a dictionary."""
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        return payload
    except Exception:  # noqa: BLE001 — malformed body is a client error
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Body must be a JSON object.",
        )


def _validate_secret(payload: dict, secret: str) -> None:
    """Validate the secret contained in the payload."""
    if str(payload.get("secret") or "") != secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bad or missing webhook secret.",
        )


def _record_alert(alert: dict) -> None:
    """Append alert to the ring buffer and trim to max size."""
    _RECENT_ALERTS.append(alert)
    # Keep only the most recent _MAX_RECENT entries
    del _RECENT_ALERTS[:-_MAX_RECENT]


async def _publish_to_redis(alert: dict) -> None:
    """Publish the alert to Redis if a client is available."""
    try:
        from app.redis_client import get_redis

        r = get_redis()
        if r is not None:
            import json as _json

            await r.publish("tradingview:alerts", _json.dumps(alert))
    except Exception as exc:  # noqa: BLE001 — receiver must not depend on Redis
        logger.debug("tradingview alert: redis publish skipped", error=str(exc))


@router.post("/tradingview")
async def receive_tradingview_alert(request: Request) -> dict:
    start_time = time.perf_counter()

    secret = _get_secret()
    _ensure_secret_enabled(secret)

    payload = await _parse_payload(request)
    _validate_secret(payload, secret)

    alert = _normalize(payload)
    _record_alert(alert)

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

    await _publish_to_redis(alert)

    return {"ok": True, "alert": alert}


@router.get("/tradingview/recent")
async def recent_tradingview_alerts(limit: int = 50) -> dict:
    """Most recent received alerts (process-local ring buffer)."""
    limit = max(1, min(limit, _MAX_RECENT))
    return {"alerts": _RECENT_ALERTS[-limit:][::-1], "count": len(_RECENT_ALERTS)}