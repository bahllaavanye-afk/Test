"""Market regime and cross-strategy correlation endpoints."""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.models.user import User
from app.ml.regime.detector import regime_monitor
from app.risk.correlation_monitor import correlation_monitor

router = APIRouter(prefix="/regime", tags=["regime"])

# --------------------------------------------------------------------------- #
# Simple in‑memory caching utilities (TTL based) to avoid recomputing expensive
# data structures on every request. The cache is deliberately lightweight and
# lives for a few seconds – enough to amortise the cost of the underlying
# calculations while keeping data reasonably fresh.
# --------------------------------------------------------------------------- #

_STATE_TTL = 5  # seconds
_CORR_TTL = 5  # seconds

_states_cache: Dict[str, Any] | None = None
_states_timestamp: float = 0.0

_corr_cache: List[List[float]] | None = None
_corr_timestamp: float = 0.0


def _cached_states() -> Dict[str, Any]:
    """Return cached regime states, recomputing only after TTL expiry."""
    global _states_cache, _states_timestamp
    now = time.time()
    if _states_cache is None or now - _states_timestamp > _STATE_TTL:
        _states_cache = regime_monitor.all_states()
        _states_timestamp = now
    return _states_cache


def _cached_correlation_matrix() -> List[List[float]]:
    """Return cached correlation matrix, recomputing only after TTL expiry."""
    global _corr_cache, _corr_timestamp
    now = time.time()
    if _corr_cache is None or now - _corr_timestamp > _CORR_TTL:
        _corr_cache = correlation_monitor.matrix_as_list()
        _corr_timestamp = now
    return _corr_cache


@router.get("/current")
async def get_current_regime(current_user: User = Depends(get_current_user)):
    """Overall market regime — aggregated across all tracked symbols.

    Returns the most common regime (bull/bear/sideways mapped from detector enums)
    and average confidence. Falls back to safe defaults when no data is available.
    """
    states = _cached_states()
    if not states:
        return {"regime": "unknown", "confidence": 0.0, "updated_at": None, "symbol_count": 0}

    # Map detector regimes → frontend‑friendly labels
    _label_map = {
        "trending": "bull",
        "mean_reverting": "sideways",
        "high_vol": "bear",
        "unknown": "unknown",
    }

    label_counts: Counter = Counter()
    confidences: List[float] = []
    latest_updated: str | None = None

    for sym_state in states.values():
        raw = sym_state.get("regime", "unknown")
        label = _label_map.get(raw, "unknown")
        label_counts[label] += 1
        confidences.append(sym_state.get("confidence", 0.0))
        updated = sym_state.get("updated_at")
        if updated and (latest_updated is None or updated > latest_updated):
            latest_updated = updated

    overall_regime = label_counts.most_common(1)[0][0]
    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    return {
        "regime": overall_regime,
        "confidence": avg_confidence,
        "updated_at": latest_updated,
        "symbol_count": len(states),
    }


@router.get("/states")
async def get_regime_states(current_user: User = Depends(get_current_user)):
    """Current regime classification for all tracked symbols."""
    return _cached_states()


@router.get("/states/{symbol}")
async def get_regime_for_symbol(symbol: str, current_user: User = Depends(get_current_user)):
    """Regime classification for a single symbol."""
    state = regime_monitor.get(symbol.upper())
    if not state:
        return {"error": f"No regime data for {symbol}. Feed price data first."}
    return state.to_dict()


@router.get("/correlation")
async def get_correlation_matrix(current_user: User = Depends(get_current_user)):
    """Live cross‑strategy correlation matrix."""
    return {
        "matrix": _cached_correlation_matrix(),
        "reduced_strategies": list(correlation_monitor._reduced),
        "recent_alerts": correlation_monitor.recent_alerts(10),
    }


@router.get("/correlation/alerts")
async def get_correlation_alerts(current_user: User = Depends(get_current_user)):
    """Recent correlation alerts (up to 50)."""
    return correlation_monitor.recent_alerts(50)