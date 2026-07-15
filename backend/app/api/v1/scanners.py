"""Scanner API — expose multi-desk stock scanner results."""
from __future__ import annotations

import json
from typing import Any, Dict, List

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
    data: Dict[str, Any] = {}


class ScanResponse(BaseModel):
    desk: str
    results: List[ScanResultOut]
    cached: bool = True


# Lazy imports for scanner classes – moved to module level to avoid repeated import overhead.
try:
    from app.tasks.stock_scanners import EquityScanner, CryptoScanner, PolymarketScanner
except Exception:  # pragma: no cover
    EquityScanner = CryptoScanner = PolymarketScanner = None  # type: ignore


# Cache scanner instances to avoid re‑instantiation on every request.
_scanner_instances: Dict[str, Any] = {}


def _get_scanner(desk: str):
    """Return a cached scanner instance for the given desk."""
    if desk in _scanner_instances:
        return _scanner_instances[desk]

    if desk == "equity":
        scanner = EquityScanner()
    elif desk == "crypto":
        scanner = CryptoScanner()
    else:
        scanner = PolymarketScanner()
    _scanner_instances[desk] = scanner
    return scanner


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
        raise HTTPException(
            status_code=400,
            detail=f"Unknown desk '{desk}'. Choose equity|crypto|polymarket",
        )

    redis = await _get_redis()
    cache_key = f"scanner:{desk}:top10"

    if not live and redis:
        try:
            raw = await redis.get(cache_key)
            if raw:
                items = json.loads(raw)
                return ScanResponse(desk=desk, results=items, cached=True)
        except Exception:
            pass

    # Live scan
    try:
        scanner = _get_scanner(desk)
        results = await scanner.scan()
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

        # Store fresh results in Redis for subsequent cached calls.
        if redis:
            try:
                await redis.set(cache_key, json.dumps([item.dict() for item in out]), ex=300)
            except Exception:
                pass

        return ScanResponse(desk=desk, results=out, cached=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=list[ScanResponse])
async def get_all_scan_results(user=Depends(get_current_user)):
    """Get cached scanner results for all three desks."""
    redis = await _get_redis()
    responses: List[ScanResponse] = []
    for desk in ("equity", "crypto", "polymarket"):
        cached_results: List[Dict[str, Any]] = []
        if redis:
            try:
                raw = await redis.get(f"scanner:{desk}:top10")
                if raw:
                    cached_results = json.loads(raw)
            except Exception:
                pass
        responses.append(ScanResponse(desk=desk, results=cached_results, cached=True))
    return responses