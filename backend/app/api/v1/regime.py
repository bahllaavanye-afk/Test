"""Market regime and cross-strategy correlation endpoints."""
import logging
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, validator

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


class CurrentRegimeResponse(BaseModel):
    """Aggregated market regime information."""

    regime: str = Field(
        ...,
        description="Overall market regime label.",
        example="bull",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average confidence of regime detection (0‑1).",
        example=0.85,
    )
    updated_at: Optional[str] = Field(
        None,
        description="ISO‑8601 timestamp of the most recent regime update.",
        example="2023-09-15T12:34:56Z",
    )
    symbol_count: int = Field(
        ...,
        ge=0,
        description="Number of symbols considered in the aggregation.",
        example=42,
    )

    @validator("updated_at")
    def validate_timestamp(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Ensure ISO format; will raise ValueError if invalid
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class ErrorResponse(BaseModel):
    """Standard error payload."""

    error: str = Field(
        ...,
        description="Human‑readable error description.",
        example="No regime data for XYZ. Feed price data first.",
    )


class RegimeStateResponse(BaseModel):
    """Regime classification for a single symbol."""

    regime: str = Field(
        ...,
        description="Regime label for the symbol.",
        example="sideways",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence of the regime detection (0‑1).",
        example=0.73,
    )
    updated_at: Optional[str] = Field(
        None,
        description="ISO‑8601 timestamp of the last update.",
        example="2023-09-15T12:34:56Z",
    )

    @validator("updated_at")
    def validate_timestamp(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    class Config:
        extra = "allow"


class CorrelationMatrixResponse(BaseModel):
    """Live cross‑strategy correlation matrix."""

    matrix: List[List[float]] = Field(
        ...,
        description="Two‑dimensional correlation matrix values.",
        example=[[1.0, 0.2], [0.2, 1.0]],
    )
    reduced_strategies: List[str] = Field(
        ...,
        description="Identifiers of strategies after dimensionality reduction.",
        example=["mean_rev_20_1.5", "time_series_momentum"],
    )
    recent_alerts: List[Dict[str, Any]] = Field(
        ...,
        description="Most recent correlation alerts.",
        example=[
            {
                "strategy": "GDX",
                "value": 0.95,
                "timestamp": "2023-09-15T12:00:00Z",
            }
        ],
    )


class AlertResponse(BaseModel):
    """Individual correlation alert."""

    class Config:
        extra = "allow"


@router.get("/current", response_model=CurrentRegimeResponse)
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
            "endpoint=get_current_regime",
            extra={
                "signal_count": 0,
                "execution_time_ms": round(elapsed_ms, 2),
                "pnl": get_current_pnl(),
            },
        )
        return CurrentRegimeResponse(
            regime="unknown",
            confidence=0.0,
            updated_at=None,
            symbol_count=0,
        )

    # Map detector regimes → frontend‑friendly labels
    _label_map = {
        "trending": "bull",
        "mean_reverting": "sideways",
        "high_vol": "bear",
        "unknown": "unknown",
    }

    label_counts: Counter = Counter()
    confidences: List[float] = []
    latest_updated: Optional[str] = None

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

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        "endpoint=get_current_regime",
        extra={
            "signal_count": len(states),
            "execution_time_ms": round(elapsed_ms, 2),
            "pnl": get_current_pnl(),
        },
    )

    return CurrentRegimeResponse(
        regime=overall_regime,
        confidence=avg_confidence,
        updated_at=latest_updated,
        symbol_count=len(states),
    )


@router.get("/states")
async def get_regime_states(current_user: User = Depends(get_current_user)):
    """Current regime classification for all tracked symbols."""
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


@router.get(
    "/states/{symbol}",
    response_model=RegimeStateResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_regime_for_symbol(symbol: str, current_user: User = Depends(get_current_user)):
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
        return ErrorResponse(error=f"No regime data for {symbol}. Feed price data first.")
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
    return RegimeStateResponse(**result)


@router.get("/correlation", response_model=CorrelationMatrixResponse)
async def get_correlation_matrix(current_user: User = Depends(get_current_user)):
    """Live cross‑strategy correlation matrix."""
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
    return CorrelationMatrixResponse(
        matrix=matrix,
        reduced_strategies=reduced,
        recent_alerts=alerts,
    )


@router.get(
    "/correlation/alerts",
    response_model=List[AlertResponse],
)
async def get_correlation_alerts(current_user: User = Depends(get_current_user)):
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
    return [AlertResponse(**a) for a in alerts]