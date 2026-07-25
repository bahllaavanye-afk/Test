"""Scanner API — expose multi-desk stock scanner results."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.deps import get_current_user

router = APIRouter(prefix="/scanners", tags=["scanners"])


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
                    items = json.loads(raw)
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

        out = [
            ScanResultOut(
                symbol=r.symbol,
                desk=r.desk,
                score=r.score,
                signals=r.signals,
                side=r.side,
                data=r.data,
            )
            for r in results[:20]
        ]
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
                    cached_results = json.loads(raw)
            except Exception:
                pass
        responses.append(ScanResponse(desk=desk, results=cached_results, cached=True))
    return responses