"""Scanner API — expose multi-desk stock scanner results."""
from __future__ import annotations

import json
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.deps import get_current_user

router = APIRouter(prefix="/scanners", tags=["scanners"])


class ScanResultOut(BaseModel):
    symbol: str
    desk: str
    score: float
    signals: List[str]
    side: str
    data: dict[str, Any] = {}


class ScanResponse(BaseModel):
    desk: str
    results: List[ScanResultOut]
    cached: bool = True


async def _get_redis():
    try:
        from app.redis_client import get_redis
        return get_redis()
    except Exception:
        return None


@router.get("/{desk}", response_model=ScanResponse)
async def get_scan_results(
    desk: Optional[str] = None,
    live: bool = Query(False, description="Re-run scanner instead of using cache"),
    user=Depends(get_current_user),
):
    """
    Get latest scanner results for a desk.
    Desks: equity, crypto, polymarket
    By default returns cached results (refreshed every 5 min by scheduler).
    Pass ?live=true to trigger an immediate re-scan (slower).
    """
    if not desk:
        raise HTTPException(status_code=400, detail="Desk parameter is required.")
    if desk not in ("equity", "crypto", "polymarket"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown desk '{desk}'. Choose equity|crypto|polymarket",
        )

    if not live:
        redis = await _get_redis()
        if redis:
            try:
                raw = await redis.get(f"scanner:{desk}:top10")
                if raw:
                    try:
                        items = json.loads(raw)
                    except json.JSONDecodeError:
                        items = []
                    if isinstance(items, list):
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

        # Guard against None or non-iterable results
        if not results:
            results = []

        out = [
            ScanResultOut(
                symbol=getattr(r, "symbol", ""),
                desk=getattr(r, "desk", desk),
                score=getattr(r, "score", 0.0),
                signals=getattr(r, "signals", []) or [],
                side=getattr(r, "side", ""),
                data=getattr(r, "data", {}) or {},
            )
            for r in results[:20] if r is not None
        ]
        return ScanResponse(desk=desk, results=out, cached=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[ScanResponse])
async def get_all_scan_results(user=Depends(get_current_user)):
    """Get cached scanner results for all three desks."""
    redis = await _get_redis()
    responses: List[ScanResponse] = []
    for desk in ("equity", "crypto", "polymarket"):
        cached_results: List[Any] = []
        if redis:
            try:
                raw = await redis.get(f"scanner:{desk}:top10")
                if raw:
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        parsed = []
                    if isinstance(parsed, list):
                        cached_results = parsed
            except Exception:
                pass
        responses.append(ScanResponse(desk=desk, results=cached_results, cached=True))
    return responses