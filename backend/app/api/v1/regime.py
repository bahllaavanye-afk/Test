"""Market regime and cross-strategy correlation endpoints."""
from collections import Counter
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User
from app.ml.regime.detector import regime_monitor
from app.risk.correlation_monitor import correlation_monitor

# Constants
ROUTER_PREFIX: str = "/regime"
ROUTER_TAGS: list[str] = ["regime"]

DEFAULT_REGIME: str = "unknown"
LABEL_MAP: dict[str, str] = {
    "trending": "bull",
    "mean_reverting": "sideways",
    "high_vol": "bear",
    "unknown": "unknown",
}
CONFIDENCE_PRECISION: int = 3

DEFAULT_ALERT_LIMIT: int = 10
MAX_ALERT_LIMIT: int = 50

router = APIRouter(prefix=ROUTER_PREFIX, tags=ROUTER_TAGS)


@router.get("/current")
async def get_current_regime(current_user: User = Depends(get_current_user)):
    """Overall market regime — aggregated across all tracked symbols.

    Returns the most common regime (bull/bear/sideways mapped from detector enums)
    and average confidence. Falls back to safe defaults when no data is available.
    """
    states = regime_monitor.all_states()
    if not states:
        return {"regime": DEFAULT_REGIME, "confidence": 0.0, "updated_at": None}

    label_counts: Counter = Counter()
    confidences: list[float] = []
    latest_updated: str | None = None

    for sym_state in states.values():
        raw = sym_state.get("regime", DEFAULT_REGIME)
        label = LABEL_MAP.get(raw, DEFAULT_REGIME)
        label_counts[label] += 1
        confidences.append(sym_state.get("confidence", 0.0))
        updated = sym_state.get("updated_at")
        if updated and (latest_updated is None or updated > latest_updated):
            latest_updated = updated

    overall_regime = label_counts.most_common(1)[0][0]
    avg_confidence = round(sum(confidences) / len(confidences), CONFIDENCE_PRECISION) if confidences else 0.0

    return {
        "regime": overall_regime,
        "confidence": avg_confidence,
        "updated_at": latest_updated,
        "symbol_count": len(states),
    }


@router.get("/states")
async def get_regime_states(current_user: User = Depends(get_current_user)):
    """Current regime classification for all tracked symbols."""
    return regime_monitor.all_states()


@router.get("/states/{symbol}")
async def get_regime_for_symbol(symbol: str, current_user: User = Depends(get_current_user)):
    state = regime_monitor.get(symbol.upper())
    if not state:
        return {"error": f"No regime data for {symbol}. Feed price data first."}
    return state.to_dict()


@router.get("/correlation")
async def get_correlation_matrix(current_user: User = Depends(get_current_user)):
    """Live cross-strategy correlation matrix."""
    return {
        "matrix": correlation_monitor.matrix_as_list(),
        "reduced_strategies": list(correlation_monitor._reduced),
        "recent_alerts": correlation_monitor.recent_alerts(DEFAULT_ALERT_LIMIT),
    }


@router.get("/correlation/alerts")
async def get_correlation_alerts(current_user: User = Depends(get_current_user)):
    return correlation_monitor.recent_alerts(MAX_ALERT_LIMIT)