"""Market regime and cross‑strategy correlation endpoints.

This module provides FastAPI routes to expose the current market regime,
per‑symbol regime states, and live cross‑strategy correlation information.
All endpoints log execution metrics and include basic error handling.
"""

import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional

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
        """Return a default P&L value when the tracker is unavailable."""
        return 0.0


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/regime", tags=["regime"])


@router.get("/current")
async def get_current_regime(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the aggregated market regime.

    The endpoint aggregates per‑symbol regime classifications from
    ``regime_monitor`` and returns the most common regime label together
    with the average confidence and the timestamp of the latest update.

    If no regime data is available, a safe default response is returned.
    """
    start_time = time.time()

    states = regime_monitor.all_states()
    if not states:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "endpoint=get_current_regime",
            extra={
                "signal_count": 0,
                "execution_time_ms": round(elapsed_ms, 2),
                "pnl": get_current_pnl(),
            },
        )
        return {"regime": "unknown", "confidence": 0.0, "updated_at": None}

    # Map detector regimes → frontend‑friendly labels
    _label_map = {
        "trending": "bull",
        "mean_reverting": "sideways",
        "high_vol": "bear",
        "unknown": "unknown",
    }

    # Use comprehensions and Counter for efficient aggregation
    label_counts: Counter[str] = Counter(
        _label_map.get(sym_state.get("regime", "unknown"), "unknown")
        for sym_state in states.values()
    )
    confidences: List[float] = [
        float(sym_state.get("confidence", 0.0)) for sym_state in states.values()
    ]
    latest_updated: Optional[str] = max(
        (
            sym_state.get("updated_at")
            for sym_state in states.values()
            if sym_state.get("updated_at")
        ),
        default=None,
    )

    overall_regime = label_counts.most_common(1)[0][0]
    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        "endpoint=get_current_regime",
        extra={
            "signal_count": len(states),
            "execution_time_ms": round(elapsed_ms, 2),
            "pnl": get_current_pnl(),
        },
    )

    return {
        "regime": overall_regime,
        "confidence": avg_confidence,
        "updated_at": latest_updated,
        "symbol_count": len(states),
    }


@router.get("/states")
async def get_regime_states(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the current regime classification for all tracked symbols."""
    start_time = time.time()
    data = regime_monitor.all_states()
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        "endpoint=get_regime_states",
        extra={
            "signal_count": len(data),
            "execution_time_ms": round(elapsed_ms, 2),
            "pnl": get_current_pnl(),
        },
    )
    return data


@router.get("/states/{symbol}")
async def get_regime_for_symbol(
    symbol: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the regime state for a specific symbol.

    If the symbol has no associated regime data, an error message is returned.
    """
    start_time = time.time()
    state = regime_monitor.get(symbol.upper())
    if not state:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "endpoint=get_regime_for_symbol",
            extra={
                "signal_count": 0,
                "execution_time_ms": round(elapsed_ms, 2),
                "pnl": get_current_pnl(),
                "symbol": symbol,
            },
        )
        return {"error": f"No regime data for {symbol}. Feed price data first."}
    result = state.to_dict()
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        "endpoint=get_regime_for_symbol",
        extra={
            "signal_count": 1,
            "execution_time_ms": round(elapsed_ms, 2),
            "pnl": get_current_pnl(),
            "symbol": symbol,
        },
    )
    return result


@router.get("/correlation")
async def get_correlation_matrix(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the live cross‑strategy correlation matrix.

    The response includes the full matrix, a list of reduced strategies,
    and the most recent alerts.
    """
    start_time = time.time()
    matrix = correlation_monitor.matrix_as_list()
    reduced = list(correlation_monitor._reduced)
    alerts = correlation_monitor.recent_alerts(10)
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        "endpoint=get_correlation_matrix",
        extra={
            "signal_count": len(matrix),
            "execution_time_ms": round(elapsed_ms, 2),
            "pnl": get_current_pnl(),
        },
    )
    return {
        "matrix": matrix,
        "reduced_strategies": reduced,
        "recent_alerts": alerts,
    }


@router.get("/correlation/alerts")
async def get_correlation_alerts(
    current_user: User = Depends(get_current_user),
) -> List[Any]:
    """Return recent cross‑strategy correlation alerts."""
    start_time = time.time()
    alerts = correlation_monitor.recent_alerts(50)
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        "endpoint=get_correlation_alerts",
        extra={
            "signal_count": len(alerts),
            "execution_time_ms": round(elapsed_ms, 2),
            "pnl": get_current_pnl(),
        },
    )
    return alerts