"""Inbound webhook receivers — TradingView alerts.

TradingView alerts can POST JSON payloads to this endpoint (charts → webhook‑IN).
The endpoint records alerts for visibility only; it does not trigger any
trading actions. Because alerts are unauthenticated third‑party input, the
shared secret is expected inside the JSON body under the key ``secret``.
If the environment variable ``TRADINGVIEW_WEBHOOK_SECRET`` is not set, the
endpoint is disabled and returns HTTP 503.

This module provides:
* A POST endpoint to receive and log TradingView alerts.
* A GET endpoint to retrieve the most recent alerts stored in a process‑local
  ring buffer.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, status

from app.utils.logging import logger

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Ring buffer of the most recent alerts (process‑local; visibility, not storage
# of record). A dead Redis must not break the receiver.
_RECENT_ALERTS: List[Dict[str, Any]] = []
_MAX_RECENT = 200

# Global counters for monitoring
_TOTAL_ALERTS: int = 0


def _normalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a free‑form TradingView alert payload.

    The function extracts a known subset of fields and coerces them into a
    consistent shape suitable for logging and downstream consumption.

    Args:
        payload: The raw JSON object received from TradingView.

    Returns:
        A dictionary containing the normalized fields:
        ``symbol``, ``side``, ``price``, ``strategy``, ``message`` and
        ``received_at``. Missing or unparsable values are represented as ``None``.
    """
    return {
        "symbol": str(payload.get("ticker") or payload.get("symbol") or "").upper()
        or None,
        "side": (str(payload.get("action") or payload.get("side") or "").lower() or None),
        "price": _float_or_none(payload.get("price") or payload.get("close")),
        "strategy": payload.get("strategy") or payload.get("indicator"),
        "message": str(payload.get("message") or payload.get("comment") or "")[:500] or None,
        "received_at": datetime.now(timezone.utc).isoformat(),
    }


def _float_or_none(v: Any) -> float | None:
    """Convert a value to ``float`` when possible.

    Args:
        v: The value to convert. ``None`` is returned unchanged.

    Returns:
        The float representation of ``v`` or ``None`` if conversion fails.
    """
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


@router.post("/tradingview")
async def receive_tradingview_alert(request: Request) -> Dict[str, Any]:
    """Receive a TradingView alert, validate it, and store a normalized copy.

    The endpoint performs the following steps:
    1. Verify that the webhook secret is configured.
    2. Parse the request body as JSON and ensure it is an object.
    3. Check the supplied secret against the configured secret.
    4. Normalize the payload and append it to the in‑memory ring buffer.
    5. Increment the global alert counter and emit structured logs.
    6. Attempt to publish the alert to Redis for downstream consumers.

    Args:
        request: The incoming FastAPI request containing the JSON payload.

    Returns:
        A JSON‑serialisable dictionary with ``ok`` set to ``True`` and the
        normalized ``alert`` payload.
    """
    start_time = time.perf_counter()

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
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Body must be a JSON object.",
        )

    if str(payload.get("secret") or "") != secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bad or missing webhook secret.",
        )

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
async def recent_tradingview_alerts(limit: int = 50) -> Dict[str, Any]:
    """Return the most recent received TradingView alerts.

    The alerts are stored in a process‑local ring buffer. The ``limit`` query
    parameter caps the number of alerts returned, bounded by the size of the
    buffer.

    Args:
        limit: Maximum number of alerts to return (default 50). The value is
            clamped between 1 and ``_MAX_RECENT``.

    Returns:
        A dictionary with two keys:
        * ``alerts`` – a list of alert dictionaries ordered from newest to oldest.
        * ``count`` – the total number of alerts currently stored in the buffer.
    """
    limit = max(1, min(limit, _MAX_RECENT))
    return {"alerts": _RECENT_ALERTS[-limit:][::-1], "count": len(_RECENT_ALERTS)}