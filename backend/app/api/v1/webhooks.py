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
from pydantic import BaseModel, Field, ValidationError, validator

from app.utils.logging import logger

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Ring buffer of the most recent alerts (process-local; visibility, not storage
# of record). A dead Redis must not break the receiver.
_RECENT_ALERTS: list[dict] = []
_MAX_RECENT = 200


class TradingViewAlert(BaseModel):
    """Schema for incoming TradingView webhook payloads."""

    secret: str = Field(
        ...,
        description="Shared secret used to authenticate the webhook request.",
        example="mysecret",
    )
    ticker: str | None = Field(
        None,
        description="Ticker symbol supplied by TradingView (e.g., 'AAPL').",
        example="AAPL",
    )
    symbol: str | None = Field(
        None,
        description="Alternative field for the symbol, if provided.",
        example="AAPL",
    )
    action: str | None = Field(
        None,
        description="Action string, often 'buy' or 'sell'.",
        example="buy",
    )
    side: str | None = Field(
        None,
        description="Side of the trade, e.g., 'long' or 'short'.",
        example="sell",
    )
    price: float | None = Field(
        None,
        description="Price at which the alert was generated.",
        example=150.23,
    )
    close: float | None = Field(
        None,
        description="Close price, used as a fallback for price.",
        example=150.0,
    )
    strategy: str | None = Field(
        None,
        description="Name of the strategy that generated the alert.",
        example="mean_rev_20_2",
    )
    indicator: str | None = Field(
        None,
        description="Alternative field for strategy/indicator name.",
        example="mean_rev_20_2",
    )
    message: str | None = Field(
        None,
        description="Human‑readable message attached to the alert.",
        example="Potential reversal signal",
    )
    comment: str | None = Field(
        None,
        description="Additional comment field, often synonymous with message.",
        example="Check volume",
    )

    class Config:
        extra = "allow"

    @validator("price", "close", pre=True)
    def _parse_float(cls, v: Any) -> float | None:
        """Coerce string or numeric inputs to float, returning None on failure."""
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None


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
        payload = TradingViewAlert(**raw_payload)
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Payload validation error: {ve.errors()}",
        )
    except Exception:  # noqa: BLE001 — malformed body is a client error
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Body must be a JSON object.",
        )

    if payload.secret != secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bad or missing webhook secret.",
        )

    alert = _normalize(payload.dict())
    _RECENT_ALERTS.append(alert)
    del _RECENT_ALERTS[:-_MAX_RECENT]
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

            await r.publish("tradingview:alerts", _json.dumps(alert))
    except Exception as exc:  # noqa: BLE001 — receiver must not depend on Redis
        logger.debug("tradingview alert: redis publish skipped", error=str(exc))

    return {"ok": True, "alert": alert}


@router.get("/tradingview/recent")
async def recent_tradingview_alerts(limit: int = 50) -> dict:
    """Most recent received alerts (process-local ring buffer)."""
    limit = max(1, min(limit, _MAX_RECENT))
    return {"alerts": _RECENT_ALERTS[-limit:][::-1], "count": len(_RECENT_ALERTS)}