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
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, validator

from app.utils.logging import logger

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Ring buffer of the most recent alerts (process-local; visibility, not storage
# of record). A dead Redis must not break the receiver.
_RECENT_ALERTS: List[Dict[str, Any]] = []
_MAX_RECENT = 200

# Global counters for monitoring
_TOTAL_ALERTS: int = 0


class TradingViewPayload(BaseModel):
    """Schema for incoming TradingView webhook payload."""

    secret: str = Field(
        ...,
        description="Shared secret for authentication.",
        example="my_super_secret",
    )
    ticker: str | None = Field(
        None,
        description="Ticker symbol, e.g., 'AAPL'.",
        example="AAPL",
    )
    symbol: str | None = Field(
        None,
        description="Alternative symbol field if ticker is not provided.",
        example="AAPL",
    )
    action: str | None = Field(
        None,
        description="Action name, typically 'buy' or 'sell'.",
        example="buy",
    )
    side: str | None = Field(
        None,
        description="Side of the trade, often duplicate of action.",
        example="sell",
    )
    price: float | None = Field(
        None,
        description="Price associated with the alert.",
        example=150.23,
    )
    close: float | None = Field(
        None,
        description="Close price, used as fallback when price is missing.",
        example=149.95,
    )
    strategy: str | None = Field(
        None,
        description="Strategy name that generated the alert.",
        example="mean_rev_20_1.5",
    )
    indicator: str | None = Field(
        None,
        description="Indicator name, alternative to strategy.",
        example="RSI",
    )
    message: str | None = Field(
        None,
        description="Human‑readable message attached to the alert.",
        example="Potential reversal detected.",
    )
    comment: str | None = Field(
        None,
        description="Alternative comment field for the alert message.",
        example="Check entry point.",
    )

    @validator("price", "close", pre=True)
    def _parse_float(cls, v: Any) -> float | None:  # noqa: N805
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError("must be a numeric value")

    @validator("ticker", "symbol", pre=True, always=True)
    def _strip_and_upper(cls, v: Any) -> str | None:  # noqa: N805
        if v is None:
            return None
        return str(v).strip().upper()


class NormalizedAlert(BaseModel):
    """Normalized representation of a TradingView alert."""

    symbol: str | None = Field(
        None,
        description="Upper‑case ticker symbol derived from payload.",
        example="AAPL",
    )
    side: str | None = Field(
        None,
        description="Side of the alert in lower case.",
        example="buy",
    )
    price: float | None = Field(
        None,
        description="Normalized price value.",
        example=150.23,
    )
    strategy: str | None = Field(
        None,
        description="Strategy or indicator that produced the alert.",
        example="mean_rev_20_1.5",
    )
    message: str | None = Field(
        None,
        description="Truncated message text (max 500 chars).",
        example="Potential reversal detected.",
    )
    received_at: datetime = Field(
        ...,
        description="Timestamp when the alert was processed.",
        example="2026-08-07T12:34:56.789Z",
    )


def _normalize(payload: TradingViewPayload) -> NormalizedAlert:
    """Best‑effort normalization of TradingView's free‑form alert JSON."""
    return NormalizedAlert(
        symbol=payload.ticker or payload.symbol,
        side=(payload.action or payload.side or None),
        price=payload.price or payload.close,
        strategy=payload.strategy or payload.indicator,
        message=(payload.message or payload.comment or None)[:500] if (payload.message or payload.comment) else None,
        received_at=datetime.now(timezone.utc),
    )


@router.post("/tradingview", response_model=Dict[str, Any])
async def receive_tradingview_alert(request: Request) -> Dict[str, Any]:
    start_time = time.perf_counter()

    secret = os.environ.get("TRADINGVIEW_WEBHOOK_SECRET", "").strip()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TradingView webhook receiver disabled — set TRADINGVIEW_WEBHOOK_SECRET.",
        )

    try:
        raw_payload = await request.json()
        if not isinstance(raw_payload, dict):
            raise ValueError("payload must be a JSON object")
    except Exception:  # noqa: BLE001 — malformed body is a client error
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Body must be a JSON object.",
        )

    # Validate payload against schema
    try:
        payload = TradingViewPayload.parse_obj(raw_payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        )

    if payload.secret != secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bad or missing webhook secret.",
        )

    alert = _normalize(payload)
    alert_dict = alert.dict()
    _RECENT_ALERTS.append(alert_dict)
    del _RECENT_ALERTS[:-_MAX_RECENT]

    # Update monitoring counters
    global _TOTAL_ALERTS
    _TOTAL_ALERTS += 1

    # Structured logging of the received alert with monitoring metrics
    exec_time = time.perf_counter() - start_time
    logger.info(
        "tradingview alert received",
        symbol=alert_dict["symbol"],
        side=alert_dict["side"],
        strategy=str(alert_dict["strategy"])[:40],
        signal_count=_TOTAL_ALERTS,
        exec_time_ms=round(exec_time * 1000, 2),
        pnl=alert_dict.get("pnl"),
    )

    # Best‑effort fan‑out to Redis subscribers (strategies/dashboards may listen).
    try:
        from app.redis_client import get_redis

        r = get_redis()
        if r is not None:
            import json as _json

            await r.publish("tradingview:alerts", _json.dumps(alert_dict))
    except Exception as exc:  # noqa: BLE001 — receiver must not depend on Redis
        logger.debug("tradingview alert: redis publish skipped", error=str(exc))

    return {"ok": True, "alert": alert_dict}


@router.get("/tradingview/recent", response_model=Dict[str, Any])
async def recent_tradingview_alerts(limit: int = 50) -> Dict[str, Any]:
    """Most recent received alerts (process‑local ring buffer)."""
    limit = max(1, min(limit, _MAX_RECENT))
    return {"alerts": _RECENT_ALERTS[-limit:][::-1], "count": len(_RECENT_ALERTS)}