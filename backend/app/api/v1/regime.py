"""Market regime and cross-strategy correlation endpoints."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional

from app.api.deps import get_current_user
from app.models.user import User
from app.ml.regime.detector import regime_monitor
from app.risk.correlation_monitor import correlation_monitor

router = APIRouter(prefix="/regime", tags=["regime"])


class RegimeResponse(BaseModel):
    """Aggregated market regime information."""

    regime: str = Field(
        ...,
        description="Aggregated market regime label (bull, bear, sideways, unknown).",
        example="bull",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average confidence of the regime detection across all symbols.",
        example=0.87,
    )
    updated_at: Optional[str] = Field(
        None,
        description="ISO‑8601 timestamp of the most recent underlying data update.",
        example="2023-09-01T12:34:56Z",
    )
    symbol_count: int = Field(
        ...,
        ge=0,
        description="Number of symbols that contributed to the aggregated regime.",
        example=128,
    )

    @validator("confidence")
    def confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v


class RegimeState(BaseModel):
    """Regime classification for a single symbol."""

    regime: str = Field(
        ...,
        description="Regime label for the symbol (bull, bear, sideways, unknown).",
        example="sideways",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence of the classification for the symbol.",
        example=0.73,
    )
    updated_at: Optional[str] = Field(
        None,
        description="ISO‑8601 timestamp of the latest regime update for the symbol.",
        example="2023-09-01T12:00:00Z",
    )

    @validator("confidence")
    def confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v

    class Config:
        extra = "allow"


class CorrelationMatrixResponse(BaseModel):
    """Live cross‑strategy correlation matrix."""

    matrix: List[List[float]] = Field(
        ...,
        description="Two‑dimensional list representing correlation coefficients between strategies.",
        example=[[1.0, 0.2], [0.2, 1.0]],
    )
    reduced_strategies: List[str] = Field(
        ...,
        description="Identifiers of the strategies that were reduced for the matrix.",
        example=["mean_rev_20_2", "trend_5"],
    )
    recent_alerts: List[Dict[str, Any]] = Field(
        ...,
        description="Most recent correlation alerts with strategy identifiers and alert details.",
        example=[{"strategy": "mean_rev_20_2", "alert": "high correlation"}],
    )


@router.get("/current", response_model=RegimeResponse)
async def get_current_regime(current_user: User = Depends(get_current_user)):
    """Overall market regime — aggregated across all tracked symbols.

    Returns the most common regime (bull/bear/sideways mapped from detector enums)
    and average confidence. Falls back to safe defaults when no data is available.
    """
    states = regime_monitor.all_states()
    if not states:
        return {
            "regime": "unknown",
            "confidence": 0.0,
            "updated_at": None,
            "symbol_count": 0,
        }

    # Map detector regimes → frontend‑friendly labels
    _label_map = {
        "trending": "bull",
        "mean_reverting": "sideways",
        "high_vol": "bear",
        "unknown": "unknown",
    }

    from collections import Counter

    label_counts: Counter = Counter()
    confidences: list[float] = []
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

    return {
        "regime": overall_regime,
        "confidence": avg_confidence,
        "updated_at": latest_updated,
        "symbol_count": len(states),
    }


@router.get("/states", response_model=Dict[str, RegimeState])
async def get_regime_states(current_user: User = Depends(get_current_user)):
    """Current regime classification for all tracked symbols."""
    # Convert raw dicts to RegimeState models for validation
    raw = regime_monitor.all_states()
    return {sym: RegimeState(**state.to_dict() if hasattr(state, "to_dict") else state) for sym, state in raw.items()}


@router.get("/states/{symbol}", response_model=RegimeState | Dict[str, str])
async def get_regime_for_symbol(symbol: str, current_user: User = Depends(get_current_user)):
    state = regime_monitor.get(symbol.upper())
    if not state:
        return {"error": f"No regime data for {symbol}. Feed price data first."}
    return RegimeState(**state.to_dict())


@router.get("/correlation", response_model=CorrelationMatrixResponse)
async def get_correlation_matrix(current_user: User = Depends(get_current_user)):
    """Live cross‑strategy correlation matrix."""
    return {
        "matrix": correlation_monitor.matrix_as_list(),
        "reduced_strategies": list(correlation_monitor._reduced),
        "recent_alerts": correlation_monitor.recent_alerts(10),
    }


@router.get("/correlation/alerts", response_model=List[Dict[str, Any]])
async def get_correlation_alerts(current_user: User = Depends(get_current_user)):
    return correlation_monitor.recent_alerts(50)