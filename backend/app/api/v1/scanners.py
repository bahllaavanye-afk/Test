"""Scanner API — expose multi-desk stock scanner results."""
from __future__ import annotations

import json
import logging
from json import JSONDecodeError
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.deps import get_current_user

router = APIRouter(prefix="/scanners", tags=["scanners"])

logger = logging.getLogger(__name__)

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


async def _get_redis() -> Optional[Any]:
    """
    Retrieve the Redis client. Returns None if the client cannot be instantiated.
    """
    try:
        from app.redis_client import get_redis
        return get_redis()
    except ImportError as exc:
        logger.error("Redis client module not found.", exc_info=exc)
        return None
    except Exception as exc:
        logger.error("Unexpected error while obtaining Redis client.", exc_info=exc)
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

    if not live:
        redis = await _get_redis()
        if redis:
            try:
                raw = await redis.get(f"scanner:{desk}:top10")
                if raw:
                    try:
                        items = json.loads(raw)
                        return ScanResponse(desk=desk, results=items, cached=True)
                    except JSONDecodeError as exc:
                        logger.error(
                            "Failed to decode cached scanner results for desk %s.", desk, exc_info=exc
                        )
            except Exception as exc:
                logger.error(
                    "Error retrieving cached scanner results from Redis for desk %s.", desk, exc_info=exc
                )

    # Live scan
    try:
        from app.tasks.stock_scanners import EquityScanner, CryptoScanner, PolymarketScanner
    except ImportError as exc:
        logger.error("Scanner modules could not be imported.", exc_info=exc)
        raise HTTPException(status_code=500, detail="Scanner implementation unavailable.") from exc
    except Exception as exc:
        logger.error("Unexpected error during scanner import.", exc_info=exc)
        raise HTTPException(status_code=500, detail="Unexpected scanner import error.") from exc

    try:
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
    except Exception as exc:
        logger.error("Live scan failed for desk %s.", desk, exc_info=exc)
        raise HTTPException(status_code=500, detail="Error executing live scan.") from exc


@router.get("/", response_model=list[ScanResponse])
async def get_all_scan_results(user=Depends(get_current_user)):
    """Get cached scanner results for all three desks."""
    redis = await _get_redis()
    responses: List[ScanResponse] = []
    for desk in ("equity", "crypto", "polymarket"):
        cached_results: List[ScanResultOut] = []
        if redis:
            try:
                raw = await redis.get(f"scanner:{desk}:top10")
                if raw:
                    try:
                        cached_results = json.loads(raw)
                    except JSONDecodeError as exc:
                        logger.error(
                            "Failed to decode cached scanner results for desk %s.", desk, exc_info=exc
                        )
            except Exception as exc:
                logger.error(
                    "Error retrieving cached scanner results from Redis for desk %s.", desk, exc_info=exc
                )
        responses.append(ScanResponse(desk=desk, results=cached_results, cached=True))
    return responses