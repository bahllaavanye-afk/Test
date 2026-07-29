"""Market regime and cross-strategy correlation endpoints."""
import logging
from collections import Counter
from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_current_user
from app.models.user import User
from app.ml.regime.detector import regime_monitor
from app.risk.correlation_monitor import correlation_monitor

router = APIRouter(prefix="/regime", tags=["regime"])
logger = logging.getLogger(__name__)


@router.get("/current")
async def get_current_regime(current_user: User = Depends(get_current_user)):
    """Overall market regime — aggregated across all tracked symbols.

    Returns the most common regime (bull/bear/sideways mapped from detector enums)
    and average confidence. Falls back to safe defaults when no data is available.
    """
    try:
        states = regime_monitor.all_states()
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to retrieve regime states",
            exc_info=True,
            extra={"user_id": getattr(current_user, "id", None)},
        )
        raise HTTPException(status_code=500, detail="Unable to fetch regime data") from exc

    if not states:
        return {"regime": "unknown", "confidence": 0.0, "updated_at": None, "symbol_count": 0}

    # Map detector regimes → frontend-friendly labels
    _label_map = {
        "trending": "bull",
        "mean_reverting": "sideways",
        "high_vol": "bear",
        "unknown": "unknown",
    }

    label_counts: Counter = Counter()
    confidences: list[float] = []
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
    try:
        return regime_monitor.all_states()
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Error fetching all regime states",
            exc_info=True,
            extra={"user_id": getattr(current_user, "id", None)},
        )
        raise HTTPException(status_code=500, detail="Unable to retrieve regime states") from exc


@router.get("/states/{symbol}")
async def get_regime_for_symbol(symbol: str, current_user: User = Depends(get_current_user)):
    try:
        state = regime_monitor.get(symbol.upper())
    except KeyError as exc:
        logger.warning(
            "Regime data missing for symbol",
            extra={"symbol": symbol, "user_id": getattr(current_user, "id", None)},
        )
        raise HTTPException(status_code=404, detail=f"No regime data for {symbol}") from exc
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Unexpected error retrieving regime for symbol",
            exc_info=True,
            extra={"symbol": symbol, "user_id": getattr(current_user, "id", None)},
        )
        raise HTTPException(status_code=500, detail="Error retrieving regime data") from exc

    if not state:
        return {"error": f"No regime data for {symbol}. Feed price data first."}
    return state.to_dict()


@router.get("/correlation")
async def get_correlation_matrix(current_user: User = Depends(get_current_user)):
    """Live cross-strategy correlation matrix."""
    try:
        matrix = correlation_monitor.matrix_as_list()
        reduced = list(correlation_monitor._reduced)
        alerts = correlation_monitor.recent_alerts(10)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to build correlation matrix",
            exc_info=True,
            extra={"user_id": getattr(current_user, "id", None)},
        )
        raise HTTPException(status_code=500, detail="Unable to fetch correlation data") from exc

    return {
        "matrix": matrix,
        "reduced_strategies": reduced,
        "recent_alerts": alerts,
    }


@router.get("/correlation/alerts")
async def get_correlation_alerts(current_user: User = Depends(get_current_user)):
    try:
        return correlation_monitor.recent_alerts(50)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Error retrieving correlation alerts",
            exc_info=True,
            extra={"user_id": getattr(current_user, "id", None)},
        )
        raise HTTPException(status_code=500, detail="Unable to fetch correlation alerts") from exc