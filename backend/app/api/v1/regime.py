"""Market regime and cross-strategy correlation endpoints."""
import logging
import time
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any

from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User
from app.ml.regime.detector import regime_monitor
from app.risk.correlation_monitor import correlation_monitor

# Optional P&L import – fallback to zero if unavailable
try:
    from app.risk.pnl_tracker import get_current_pnl  # type: ignore
except Exception:  # pragma: no cover
    def get_current_pnl() -> float:
        return 0.0

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/regime", tags=["regime"])

_LABEL_MAP = {
    "trending": "bull",
    "mean_reverting": "sideways",
    "high_vol": "bear",
    "unknown": "unknown",
}


def _map_label(regime: str) -> str:
    """Map detector regime to frontend-friendly label."""
    return _LABEL_MAP.get(regime, "unknown")


def _compute_aggregates(
    states: Dict[str, Any],
) -> Tuple[str, float, Optional[str]]:
    """
    Compute overall regime, average confidence, and most recent update timestamp.

    Returns:
        overall_regime: The most common mapped regime.
        avg_confidence: Average confidence rounded to three decimals.
        latest_updated: ISO timestamp of the most recent update, if any.
    """
    label_counts: Counter = Counter(
        _map_label(sym_state.get("regime", "unknown")) for sym_state in states.values()
    )
    confidences: List[float] = [
        float(sym_state.get("confidence", 0.0)) for sym_state in states.values()
    ]
    latest_updated: Optional[str] = max(
        (sym_state.get("updated_at") for sym_state in states.values() if sym_state.get("updated_at")),
        default=None,
    )
    overall_regime = label_counts.most_common(1)[0][0] if label_counts else "unknown"
    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    return overall_regime, avg_confidence, latest_updated


def _log_endpoint(endpoint: str, signal_count: int, start_time: float) -> None:
    """Log execution details for an endpoint."""
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        f"endpoint={endpoint}",
        extra={
            "signal_count": signal_count,
            "execution_time_ms": round(elapsed_ms, 2),
            "pnl": get_current_pnl(),
        },
    )


@router.get("/current")
async def get_current_regime(current_user: User = Depends(get_current_user)):
    """Overall market regime — aggregated across all tracked symbols.

    Returns the most common regime (bull/bear/sideways mapped from detector enums)
    and average confidence. Falls back to safe defaults when no data is available.
    """
    start_time = time.time()
    states = regime_monitor.all_states()
    if not states:
        _log_endpoint("get_current_regime", 0, start_time)
        return {"regime": "unknown", "confidence": 0.0, "updated_at": None}

    overall_regime, avg_confidence, latest_updated = _compute_aggregates(states)
    _log_endpoint("get_current_regime", len(states), start_time)

    return {
        "regime": overall_regime,
        "confidence": avg_confidence,
        "updated_at": latest_updated,
        "symbol_count": len(states),
    }


@router.get("/states")
async def get_regime_states(current_user: User = Depends(get_current_user)):
    """Current regime classification for all tracked symbols."""
    start_time = time.time()
    data = regime_monitor.all_states()
    _log_endpoint("get_regime_states", len(data), start_time)
    return data


@router.get("/states/{symbol}")
async def get_regime_for_symbol(symbol: str, current_user: User = Depends(get_current_user)):
    start_time = time.time()
    state = regime_monitor.get(symbol.upper())
    if not state:
        _log_endpoint("get_regime_for_symbol", 0, start_time)
        return {"error": f"No regime data for {symbol}. Feed price data first."}
    result = state.to_dict()
    _log_endpoint("get_regime_for_symbol", 1, start_time)
    return result


@router.get("/correlation")
async def get_correlation_matrix(current_user: User = Depends(get_current_user)):
    """Live cross-strategy correlation matrix."""
    start_time = time.time()
    matrix = correlation_monitor.matrix_as_list()
    reduced = list(correlation_monitor._reduced)
    alerts = correlation_monitor.recent_alerts(10)
    _log_endpoint("get_correlation_matrix", len(matrix), start_time)
    return {
        "matrix": matrix,
        "reduced_strategies": reduced,
        "recent_alerts": alerts,
    }


@router.get("/correlation/alerts")
async def get_correlation_alerts(current_user: User = Depends(get_current_user)):
    start_time = time.time()
    alerts = correlation_monitor.recent_alerts(50)
    _log_endpoint("get_correlation_alerts", len(alerts), start_time)
    return alerts