"""API endpoints for market regime information and cross‑strategy correlation.

Provides:
- Current aggregated market regime.
- Per‑symbol regime states.
- Live correlation matrix and recent alerts.
"""

from collections import Counter
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.ml.regime.detector import regime_monitor
from app.models.user import User
from app.risk.correlation_monitor import correlation_monitor

router = APIRouter(prefix="/regime", tags=["regime"])

# Mapping from detector regime identifiers to frontend‑friendly labels
_LABEL_MAP: Dict[str, str] = {
    "trending": "bull",
    "mean_reverting": "sideways",
    "high_vol": "bear",
    "unknown": "unknown",
}


@router.get("/current")
async def get_current_regime(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Return the overall market regime aggregated across all tracked symbols.

    The response includes:
    - ``regime``: The most common regime label (bull, bear, sideways, unknown).
    - ``confidence``: Average confidence score across symbols, rounded to three decimals.
    - ``updated_at``: Timestamp of the most recent regime update.
    - ``symbol_count``: Number of symbols with regime data.

    If no regime data is available, safe default values are returned.
    """
    states = regime_monitor.all_states()
    if not states:
        return {"regime": "unknown", "confidence": 0.0, "updated_at": None}

    label_counts: Counter[str] = Counter()
    confidences: List[float] = []
    latest_updated: Optional[str] = None

    for sym_state in states.values():
        raw = sym_state.get("regime", "unknown")
        label = _LABEL_MAP.get(raw, "unknown")
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
async def get_regime_states(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Return the current regime classification for all tracked symbols."""
    return regime_monitor.all_states()


@router.get("/states/{symbol}")
async def get_regime_for_symbol(
    symbol: str, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Return the regime state for a specific symbol.

    Args:
        symbol: Ticker symbol (case‑insensitive).

    Returns:
        A dictionary representation of the regime state, or an error message if
        no data is available for the requested symbol.
    """
    state = regime_monitor.get(symbol.upper())
    if not state:
        return {"error": f"No regime data for {symbol}. Feed price data first."}
    return state.to_dict()


@router.get("/correlation")
async def get_correlation_matrix(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Return the live cross‑strategy correlation matrix and recent alerts."""
    return {
        "matrix": correlation_monitor.matrix_as_list(),
        "reduced_strategies": list(correlation_monitor._reduced),
        "recent_alerts": correlation_monitor.recent_alerts(10),
    }


@router.get("/correlation/alerts")
async def get_correlation_alerts(current_user: User = Depends(get_current_user)) -> List[Dict[str, Any]]:
    """Return recent correlation alerts (up to 50)."""
    return correlation_monitor.recent_alerts(50)