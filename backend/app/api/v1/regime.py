"""Market regime and cross-strategy correlation endpoints."""
import logging
import time
from collections import Counter

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

# Constants
PREFIX = "/regime"
TAG = "regime"

ENDPOINT_GET_CURRENT = "endpoint=get_current_regime"
ENDPOINT_GET_STATES = "endpoint=get_regime_states"
ENDPOINT_GET_SYMBOL = "endpoint=get_regime_for_symbol"
ENDPOINT_GET_CORRELATION = "endpoint=get_correlation_matrix"
ENDPOINT_GET_ALERTS = "endpoint=get_correlation_alerts"

LOGGER_KEY_SIGNAL_COUNT = "signal_count"
LOGGER_KEY_EXEC_TIME_MS = "execution_time_ms"
LOGGER_KEY_PNL = "pnl"
LOGGER_KEY_SYMBOL = "symbol"

DEFAULT_REGIME = "unknown"
DEFAULT_CONFIDENCE = 0.0

LABEL_MAP = {
    "trending": "bull",
    "mean_reverting": "sideways",
    "high_vol": "bear",
    "unknown": "unknown",
}

ROUND_MS_PLACES = 2
ROUND_CONF_PLACES = 3

ALERT_LIMIT_DEFAULT = 10
ALERT_LIMIT_MAX = 50

logger = logging.getLogger(__name__)

router = APIRouter(prefix=PREFIX, tags=[TAG])


@router.get("/current")
async def get_current_regime(current_user: User = Depends(get_current_user)):
    """Overall market regime — aggregated across all tracked symbols.

    Returns the most common regime (bull/bear/sideways mapped from detector enums)
    and average confidence. Falls back to safe defaults when no data is available.
    """
    start_time = time.time()

    states = regime_monitor.all_states()
    if not states:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            ENDPOINT_GET_CURRENT,
            extra={
                LOGGER_KEY_SIGNAL_COUNT: 0,
                LOGGER_KEY_EXEC_TIME_MS: round(elapsed_ms, ROUND_MS_PLACES),
                LOGGER_KEY_PNL: get_current_pnl(),
            },
        )
        return {"regime": DEFAULT_REGIME, "confidence": DEFAULT_CONFIDENCE, "updated_at": None}

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
    avg_confidence = round(sum(confidences) / len(confidences), ROUND_CONF_PLACES) if confidences else DEFAULT_CONFIDENCE

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        ENDPOINT_GET_CURRENT,
        extra={
            LOGGER_KEY_SIGNAL_COUNT: len(states),
            LOGGER_KEY_EXEC_TIME_MS: round(elapsed_ms, ROUND_MS_PLACES),
            LOGGER_KEY_PNL: get_current_pnl(),
        },
    )

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
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        ENDPOINT_GET_STATES,
        extra={
            LOGGER_KEY_SIGNAL_COUNT: len(data),
            LOGGER_KEY_EXEC_TIME_MS: round(elapsed_ms, ROUND_MS_PLACES),
            LOGGER_KEY_PNL: get_current_pnl(),
        },
    )
    return data


@router.get("/states/{symbol}")
async def get_regime_for_symbol(symbol: str, current_user: User = Depends(get_current_user)):
    start_time = time.time()
    state = regime_monitor.get(symbol.upper())
    if not state:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            ENDPOINT_GET_SYMBOL,
            extra={
                LOGGER_KEY_SIGNAL_COUNT: 0,
                LOGGER_KEY_EXEC_TIME_MS: round(elapsed_ms, ROUND_MS_PLACES),
                LOGGER_KEY_PNL: get_current_pnl(),
                LOGGER_KEY_SYMBOL: symbol,
            },
        )
        return {"error": f"No regime data for {symbol}. Feed price data first."}
    result = state.to_dict()
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        ENDPOINT_GET_SYMBOL,
        extra={
            LOGGER_KEY_SIGNAL_COUNT: 1,
            LOGGER_KEY_EXEC_TIME_MS: round(elapsed_ms, ROUND_MS_PLACES),
            LOGGER_KEY_PNL: get_current_pnl(),
            LOGGER_KEY_SYMBOL: symbol,
        },
    )
    return result


@router.get("/correlation")
async def get_correlation_matrix(current_user: User = Depends(get_current_user)):
    """Live cross-strategy correlation matrix."""
    start_time = time.time()
    matrix = correlation_monitor.matrix_as_list()
    reduced = list(correlation_monitor._reduced)
    alerts = correlation_monitor.recent_alerts(ALERT_LIMIT_DEFAULT)
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        ENDPOINT_GET_CORRELATION,
        extra={
            LOGGER_KEY_SIGNAL_COUNT: len(matrix),
            LOGGER_KEY_EXEC_TIME_MS: round(elapsed_ms, ROUND_MS_PLACES),
            LOGGER_KEY_PNL: get_current_pnl(),
        },
    )
    return {
        "matrix": matrix,
        "reduced_strategies": reduced,
        "recent_alerts": alerts,
    }


@router.get("/correlation/alerts")
async def get_correlation_alerts(current_user: User = Depends(get_current_user)):
    start_time = time.time()
    alerts = correlation_monitor.recent_alerts(ALERT_LIMIT_MAX)
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        ENDPOINT_GET_ALERTS,
        extra={
            LOGGER_KEY_SIGNAL_COUNT: len(alerts),
            LOGGER_KEY_EXEC_TIME_MS: round(elapsed_ms, ROUND_MS_PLACES),
            LOGGER_KEY_PNL: get_current_pnl(),
        },
    )
    return alerts