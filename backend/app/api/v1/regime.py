"""Market regime and cross-strategy correlation endpoints."""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, validator

from app.api.deps import get_current_user
from app.models.user import User
from app.ml.regime.detector import regime_monitor
from app.risk.correlation_monitor import correlation_monitor

router = APIRouter(prefix="/regime", tags=["regime"])


class CurrentRegimeResponse(BaseModel):
    """Aggregated market regime information."""

    regime: str = Field(
        ...,
        description="Overall market regime label mapped to frontend-friendly terms.",
        example="bull",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average confidence across symbols (0 – 1).",
        example=0.87,
    )
    updated_at: Optional[str] = Field(
        None,
        description="ISO‑8601 timestamp of the most recent regime update.",
        example="2023-09-01T12:34:56Z",
    )
    symbol_count: int = Field(
        ...,
        ge=0,
        description="Number of symbols that contributed to the aggregated regime.",
        example=124,
    )

    @validator("confidence", pre=True)
    def round_confidence(cls, v: float) -> float:  # noqa: D401
        """Round confidence to three decimal places."""
        return round(v, 3)


class CorrelationResponse(BaseModel):
    """Live cross‑strategy correlation data."""

    matrix: List[List[float]] = Field(
        ...,
        description="Correlation matrix where each inner list represents a row.",
        example=[[1.0, 0.2], [0.2, 1.0]],
    )
    reduced_strategies: List[str] = Field(
        ...,
        description="List of strategy identifiers retained after dimensionality reduction.",
        example=["mean_rev_20_1.5", "trend_50_2.0"],
    )
    recent_alerts: List[Dict[str, Any]] = Field(
        ...,
        description="Most recent correlation alerts with their metadata.",
        example=[{"strategy": "mean_rev_20_1.5", "alert": "high_corr", "timestamp": "2023-09-01T12:00:00Z"}],
    )


@router.get("/current", response_model=CurrentRegimeResponse)
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

    # Map detector regimes → frontend-friendly labels
    _label_map = {
        "trending": "bull",
        "mean_reverting": "sideways",
        "high_vol": "bear",
        "unknown": "unknown",
    }

    from collections import Counter

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
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

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


@router.get("/correlation", response_model=CorrelationResponse)
async def get_correlation_matrix(current_user: User = Depends(get_current_user)):
    """Live cross‑strategy correlation matrix."""
    return {
        "matrix": correlation_monitor.matrix_as_list(),
        "reduced_strategies": list(correlation_monitor._reduced),
        "recent_alerts": correlation_monitor.recent_alerts(10),
    }


@router.get("/correlation/alerts")
async def get_correlation_alerts(current_user: User = Depends(get_current_user)):
    return correlation_monitor.recent_alerts(50)