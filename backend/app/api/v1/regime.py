"""Market regime and cross-strategy correlation endpoints."""

# Constants
ROUTER_PREFIX = "/regime"
ROUTER_TAGS = ["regime"]

ENDPOINT_CURRENT = "/current"
ENDPOINT_STATES = "/states"
ENDPOINT_STATES_SYMBOL = "/states/{symbol}"
ENDPOINT_CORRELATION = "/correlation"
ENDPOINT_CORRELATION_ALERTS = "/correlation/alerts"

LABEL_MAP = {
    "trending": "bull",
    "mean_reverting": "sideways",
    "high_vol": "bear",
    "unknown": "unknown",
}

DEFAULT_REGIME = "unknown"
DEFAULT_CONFIDENCE = 0.0
DEFAULT_UPDATED_AT = None

ERROR_NO_DATA_TEMPLATE = "No regime data for {symbol}. Feed price data first."

RESPONSE_KEY_REGIME = "regime"
RESPONSE_KEY_CONFIDENCE = "confidence"
RESPONSE_KEY_UPDATED_AT = "updated_at"
RESPONSE_KEY_SYMBOL_COUNT = "symbol_count"
RESPONSE_KEY_MATRIX = "matrix"
RESPONSE_KEY_REDUCED_STRATEGIES = "reduced_strategies"
RESPONSE_KEY_RECENT_ALERTS = "recent_alerts"

ALERTS_LIMIT_DEFAULT = 10
ALERTS_LIMIT_MAX = 50
ROUND_PRECISION = 3

# Imports
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User
from app.ml.regime.detector import regime_monitor
from app.risk.correlation_monitor import correlation_monitor
from collections import Counter

router = APIRouter(prefix=ROUTER_PREFIX, tags=ROUTER_TAGS)


@router.get(ENDPOINT_CURRENT)
async def get_current_regime(current_user: User = Depends(get_current_user)):
    """Overall market regime — aggregated across all tracked symbols.

    Returns the most common regime (bull/bear/sideways mapped from detector enums)
    and average confidence. Falls back to safe defaults when no data is available.
    """
    states = regime_monitor.all_states()
    if not states:
        return {
            RESPONSE_KEY_REGIME: DEFAULT_REGIME,
            RESPONSE_KEY_CONFIDENCE: DEFAULT_CONFIDENCE,
            RESPONSE_KEY_UPDATED_AT: DEFAULT_UPDATED_AT,
        }

    label_counts: Counter = Counter()
    confidences: list[float] = []
    latest_updated: str | None = None

    for sym_state in states.values():
        raw = sym_state.get("regime", DEFAULT_REGIME)
        label = LABEL_MAP.get(raw, DEFAULT_REGIME)
        label_counts[label] += 1
        confidences.append(sym_state.get("confidence", DEFAULT_CONFIDENCE))
        updated = sym_state.get("updated_at")
        if updated and (latest_updated is None or updated > latest_updated):
            latest_updated = updated

    overall_regime = label_counts.most_common(1)[0][0]
    avg_confidence = round(
        sum(confidences) / len(confidences), ROUND_PRECISION
    ) if confidences else DEFAULT_CONFIDENCE

    return {
        RESPONSE_KEY_REGIME: overall_regime,
        RESPONSE_KEY_CONFIDENCE: avg_confidence,
        RESPONSE_KEY_UPDATED_AT: latest_updated,
        RESPONSE_KEY_SYMBOL_COUNT: len(states),
    }


@router.get(ENDPOINT_STATES)
async def get_regime_states(current_user: User = Depends(get_current_user)):
    """Current regime classification for all tracked symbols."""
    return regime_monitor.all_states()


@router.get(ENDPOINT_STATES_SYMBOL)
async def get_regime_for_symbol(symbol: str, current_user: User = Depends(get_current_user)):
    state = regime_monitor.get(symbol.upper())
    if not state:
        return {"error": ERROR_NO_DATA_TEMPLATE.format(symbol=symbol)}
    return state.to_dict()


@router.get(ENDPOINT_CORRELATION)
async def get_correlation_matrix(current_user: User = Depends(get_current_user)):
    """Live cross-strategy correlation matrix."""
    return {
        RESPONSE_KEY_MATRIX: correlation_monitor.matrix_as_list(),
        RESPONSE_KEY_REDUCED_STRATEGIES: list(correlation_monitor._reduced),
        RESPONSE_KEY_RECENT_ALERTS: correlation_monitor.recent_alerts(ALERTS_LIMIT_DEFAULT),
    }


@router.get(ENDPOINT_CORRELATION_ALERTS)
async def get_correlation_alerts(current_user: User = Depends(get_current_user)):
    return correlation_monitor.recent_alerts(ALERTS_LIMIT_MAX)