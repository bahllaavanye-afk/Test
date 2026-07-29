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
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, validator

from app.utils.logging import logger

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Ring buffer of the most recent alerts (process‑local; visibility, not storage
# of record). A dead Redis must not break the receiver.
_RECENT_ALERTS: list[dict] = []
_MAX_RECENT = 200


class TradingViewAlertPayload(BaseModel):
    """Schema for incoming TradingView webhook payloads.

    The payload is intentionally permissive – extra fields are ignored.
    """
    secret: str = Field(
        ...,
        description="Shared secret used to authenticate the webhook request.",
        example="my_super_secret",
    )
    ticker: Optional[str] = Field(
        None,
        description="Primary ticker symbol provided by TradingView.",
        example="AAPL",
    )
    symbol: Optional[str] = Field(
        None,
        description="Alternative symbol field (used if `ticker` is absent).",
        example="AAPL",
    )
    action: Optional[str] = Field(
        None,
        description="Action side, e.g., 'buy' or 'sell'.",
        example="buy",
    )
    side: Optional[str] = Field(
        None,
        description="Side alias – alternative to `action`.",
        example="sell",
    )
    price: Optional[float] = Field(
        None,
        description="Trade price supplied by the alert.",
        example=150.23,
    )
    close: Optional[float] = Field(
        None,
        description="Close price – fallback if `price` is missing.",
        example=150.0,
    )
    strategy: Optional[str] = Field(
        None,
        description="Strategy name that generated the alert.",
        example="mean_rev_20_2",
    )
    indicator: Optional[str] = Field(
        None,
        description="Indicator name – alternative to `strategy`.",
        example="my_indicator",
    )
    message: Optional[str] = Field(
        None,
        description="Human‑readable message attached to the alert.",
        example="Entry signal",
    )
    comment: Optional[str] = Field(
        None,
        description="Comment field – alternative to `message`.",
        example="Entry signal",
    )

    @validator("price", "close", pre=True)
    def _coerce_float(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError("must be a numeric value")

    class Config:
        extra = "allow"


class TradingViewAlertNormalized(BaseModel):
    """Normalized representation stored in the recent‑alerts ring buffer."""
    symbol: Optional[str] = Field(
        ...,
        description="Upper‑case ticker symbol.",
        example="AAPL",
    )
    side: Optional[str] = Field(
        ...,
        description="Trade side, lower‑cased (e.g., 'buy' or 'sell').",
        example="buy",
    )
    price: Optional[float] = Field(
        ...,
        description="Alert price.",
        example=150.23,
    )
    strategy: Optional[str] = Field(
        ...,
        description="Strategy or indicator that produced the alert.",
        example="mean_rev_20_2",
    )
    message: Optional[str] = Field(
        ...,
        description="Trimmed alert message (max 500 characters).",
        example="Entry signal",
    )
    received_at: datetime = Field(
        ...,
        description="UTC timestamp when the alert was received.",
        example="2023-01-01T12:00:00Z",
    )

    @validator("symbol", pre=True)
    def _upper_symbol(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if isinstance(v, str) else v

    @validator("side", pre=True)
    def _lower_side(cls, v: Optional[str]) -> Optional[str]:
        return v.lower() if isinstance(v, str) else v


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
async def receive_tradingview_alert(payload: TradingViewAlertPayload) -> dict:
    secret = os.environ.get("TRADINGVIEW_WEBHOOK_SECRET", "").strip()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TradingView webhook receiver disabled — set TRADINGVIEW_WEBHOOK_SECRET.",
        )

    if payload.secret != secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bad or missing webhook secret.",
        )

    alert_dict = _normalize(payload.dict())
    # Validate normalized data against the Pydantic model to guarantee schema.
    alert = TradingViewAlertNormalized(**alert_dict).dict()

    _RECENT_ALERTS.append(alert)
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
async def recent_tradingview_alerts(limit: int = 50) -> dict:
    """Most recent received alerts (process‑local ring buffer)."""
    limit = max(1, min(limit, _MAX_RECENT))
    return {"alerts": _RECENT_ALERTS[-limit:][::-1], "count": len(_RECENT_ALERTS)}