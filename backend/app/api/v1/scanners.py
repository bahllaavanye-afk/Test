"""Scanner API — expose multi-desk stock scanner results."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.deps import get_current_user

router = APIRouter(prefix="/scanners", tags=["scanners"])

# The scanners and this schema disagreed about BOTH fields, so every non-empty
# scan result raised a Pydantic ValidationError and the route answered 500.
# It was invisible: an anonymous probe gets 401, and the module had 0% test
# coverage, so nothing ever executed the serialisation path with real rows.
#
#   score  producers emit `min(score, 100)` — a 0-100 scale — against a schema
#          declaring `ge=0.0, le=1.0`
#   side   producers emit long / short / long_yes / long_no; the validator
#          allows only {buy, sell, neutral, none}
#
# Normalising here rather than changing the scanners: their 0-100 score and
# long/short vocabulary are also written to the Redis cache and consumed
# elsewhere, so the boundary is the right place to translate. It also keeps the
# documented external contract ("score normalized between 0 and 1") true, which
# it has never actually been.
_SIDE_ALIASES = {
    "long": "buy",
    "long_yes": "buy",
    "buy": "buy",
    "short": "sell",
    "long_no": "sell",
    "sell": "sell",
    "neutral": "neutral",
    "none": "none",
    "": "none",
}

_SCORE_SCALE = 100.0


def _normalise_scan_item(item: Any) -> dict:
    """Convert a ScanResult (or its cached dict form) into schema-valid fields."""
    get = item.get if isinstance(item, dict) else lambda k, d=None: getattr(item, k, d)

    try:
        raw_score = float(get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        raw_score = 0.0
    # NaN must fail SAFE. `min(1.0, nan)` returns 1.0 in Python (every
    # comparison with NaN is False), so a malformed score would otherwise
    # surface as MAXIMUM confidence on a ranking signal — the worst possible
    # direction to round in.
    if raw_score != raw_score or raw_score in (float("inf"), float("-inf")):
        raw_score = 0.0
    # Cached rows written after this fix are already 0-1; rescale only when the
    # value is clearly on the 0-100 scale, so both forms round-trip safely.
    score = raw_score / _SCORE_SCALE if raw_score > 1.0 else raw_score

    raw_side = str(get("side", "none") or "none").lower()

    return {
        "symbol": str(get("symbol", "") or ""),
        "desk": str(get("desk", "") or ""),
        "score": max(0.0, min(1.0, score)),
        "signals": list(get("signals", []) or []),
        "side": _SIDE_ALIASES.get(raw_side, "none"),
        "data": dict(get("data", {}) or {}),
    }


def _normalise_scan_items(raw_items: Any) -> list[dict]:
    """Normalise a batch and DROP rows a scanner produced nothing for.

    `ScanResultOut.validate_signals` rejects an empty `signals` list, and both
    the equity and crypto scanners return a ScanResult unconditionally — so a
    symbol where no condition fired arrives with `signals=[]`, `score=0`,
    `side="neutral"`. One of those makes the whole response a 500.

    That is what took `/api/v1/scanners/crypto` down: not a bad row, but an
    EMPTY one. It stayed hidden while the crypto scanner was starved of bars
    and returned nothing at all; the moment the bars fix restored its universe,
    it started emitting signal-less rows and the endpoint began failing.

    Dropping is right rather than inventing a placeholder signal name: a row
    with no signals, zero score and a neutral side is the scanner saying
    "nothing here", so it carries no information and would only dilute a
    ranked list. Applied on the READ path as well as the producer, because
    Redis rows written before this fix outlive the deploy.
    """
    out: list[dict] = []
    for item in raw_items or []:
        norm = _normalise_scan_item(item)
        if norm["signals"]:
            out.append(norm)
    return out


class ScanResultOut(BaseModel):
    symbol: str = Field(..., description="Ticker symbol of the instrument.", json_schema_extra={"example": "AAPL"})
    desk: str = Field(..., description="Desk name that generated the signal.", json_schema_extra={"example": "equity"})
    score: float = Field(
        ...,
        description="Confidence score of the signal, normalized between 0 and 1.",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.87},
    )
    signals: List[str] = Field(
        ...,
        description="List of signal identifiers that contributed to the score.",
        json_schema_extra={"example": ["mean_rev_20_2", "vol_breakout"]},
    )
    side: str = Field(
        ...,
        description="Suggested position side.",
        json_schema_extra={"example": "buy"},
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional data returned by the scanner.",
        json_schema_extra={"example": {"ma20": 150.3, "volume": 2_500_000}},
    )

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        allowed = {"buy", "sell", "neutral", "none"}
        if v.lower() not in allowed:
            raise ValueError(f"side must be one of {allowed}")
        return v.lower()

    @field_validator("signals")
    @classmethod
    def validate_signals(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("signals list cannot be empty")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "AAPL",
                "desk": "equity",
                "score": 0.87,
                "signals": ["mean_rev_20_2", "vol_breakout"],
                "side": "buy",
                "data": {"ma20": 150.3, "volume": 2500000},
            }
        }
    )


class ScanResponse(BaseModel):
    desk: str = Field(..., description="Desk identifier.", json_schema_extra={"example": "equity"})
    results: List[ScanResultOut] = Field(..., description="List of scan results for the desk.")
    cached: bool = Field(
        True,
        description="Indicates whether the results were retrieved from cache.",
        json_schema_extra={"example": True},
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "desk": "equity",
                "results": [
                    {
                        "symbol": "AAPL",
                        "desk": "equity",
                        "score": 0.87,
                        "signals": ["mean_rev_20_2", "vol_breakout"],
                        "side": "buy",
                        "data": {"ma20": 150.3, "volume": 2500000},
                    }
                ],
                "cached": False,
            }
        }
    )


async def _get_redis():
    try:
        from app.redis_client import get_redis

        return get_redis()
    except Exception:
        return None


@router.get("/{desk}", response_model=ScanResponse)
async def get_scan_results(
    desk: str,
    live: bool = Query(False, description="Re-run scanner instead of using cache"),
    user=Depends(get_current_user),
):
    """
    Get latest scanner results for a desk.
    Desks: equity, crypto, polymarket
    By default returns cached results (refreshed every 5 min by scheduler).
    Pass ?live=true to trigger an immediate re-scan (slower).
    """
    if desk not in ("equity", "crypto", "polymarket"):
        raise HTTPException(status_code=400, detail=f"Unknown desk '{desk}'. Choose equity|crypto|polymarket")

    if not live:
        redis = await _get_redis()
        if redis:
            try:
                raw = await redis.get(f"scanner:{desk}:top10")
                if raw:
                    items = _normalise_scan_items(json.loads(raw))
                    return ScanResponse(desk=desk, results=items, cached=True)
            except Exception:
                pass

    # Live scan
    try:
        from app.tasks.stock_scanners import EquityScanner, CryptoScanner, PolymarketScanner

        if desk == "equity":
            results = await EquityScanner().scan()
        elif desk == "crypto":
            results = await CryptoScanner().scan()
        else:
            results = await PolymarketScanner().scan()

        out = [ScanResultOut(**r) for r in _normalise_scan_items(results[:20])]
        return ScanResponse(desk=desk, results=out, cached=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=list[ScanResponse])
async def get_all_scan_results(user=Depends(get_current_user)):
    """Get cached scanner results for all three desks."""
    redis = await _get_redis()
    responses = []
    for desk in ("equity", "crypto", "polymarket"):
        cached_results = []
        if redis:
            try:
                raw = await redis.get(f"scanner:{desk}:top10")
                if raw:
                    cached_results = _normalise_scan_items(json.loads(raw))
            except Exception:
                pass
        responses.append(ScanResponse(desk=desk, results=cached_results, cached=True))
    return responses