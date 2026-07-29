"""Market regime and cross‑strategy correlation endpoints."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, validator

from app.api.deps import get_current_user
from app.models.user import User
from app.ml.regime.detector import regime_monitor
from app.risk.correlation_monitor import correlation_monitor

router = APIRouter(prefix="/regime", tags=["regime"])


class CurrentRegimeResponse(BaseModel):
    """Aggregated market regime across all tracked symbols."""

    regime: str = Field(
        ...,
        description="Overall market regime (bull, bear, sideways, unknown).",
        example="bull",
    )
    confidence: float = Field(
        ...,
        description="Average confidence of the underlying detector (0‑1).",
        example=0.872,
    )
    updated_at: Optional[str] = Field(
        None,
        description="ISO‑8601 timestamp of the most recent regime update.",
        example="2024-09-12T14:23:45Z",
    )
    symbol_count: int = Field(
        ...,
        description="Number of symbols that contributed to the aggregated regime.",
        example=42,
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
        description="Detected regime for the symbol (bull, bear, sideways, unknown).",
        example="sideways",
    )
    confidence: float = Field(
        ...,
        description="Confidence of the detector for this symbol (0‑1).",
        example=0.645,
    )
    updated_at: Optional[str] = Field(
        None,
        description="ISO‑8601 timestamp of the last update for this symbol.",
        example="2024-09-12T14:20:01Z",
    )

    @validator("confidence")
    def confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v


class CorrelationMatrixResponse(BaseModel):
    """Live cross‑strategy correlation matrix."""

    matrix: List[List[float]] = Field(
        ...,
        description="Correlation coefficients between strategies (row‑major).",
        example=[[1.0, 0.23, -0.11], [0.23, 1.0, 0.05], [-0.11, 0.05, 1.0]],
    )
    reduced_strategies: List[str] = Field(
        ...,
        description="Identifiers of the strategies represented in the reduced matrix.",
        example=["mean_rev_20_2", "trend_50_200", "volatility_breakout"],
    )
    recent_alerts: List[dict] = Field(
        ...,
        description="Most recent correlation alerts with payload details.",
        example=[{"strategy": "mean_rev_20_2", "alert": "high_corr", "value": 0.92}],
    )


class AlertsResponse(BaseModel):
    """List of recent correlation alerts."""

    alerts: List[dict] = Field(
        ...,
        description="Correlation alerts ordered from newest to oldest.",
        example=[{"strategy": "trend_50_200", "alert": "low_corr", "value": -0.45}],
    )


@router.get("/current", response_model=CurrentRegimeResponse)
async def get_current_regime(current_user: User = Depends(get_current_user)):
    """Overall market regime — aggregated across all tracked symbols.

    Returns the most common regime (bull/bear/sideways) mapped from detector enums
    and the average confidence. Falls back to safe defaults when no data is available.
    """
    states = regime_monitor.all_states()
    if not states:
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

    return CurrentRegimeResponse(
        regime=overall_regime,
        confidence=avg_confidence,
        updated_at=latest_updated,
        symbol_count=len(states),
    )


@router.get("/states", response_model=Dict[str, RegimeState])
async def get_regime_states(current_user: User = Depends(get_current_user)):
    """Current regime classification for all tracked symbols."""
    raw_states = regime_monitor.all_states()
    # Convert raw dicts to RegimeState models; unexpected keys are ignored.
    return {
        symbol: RegimeState(**state.to_dict() if hasattr(state, "to_dict") else state)
        for symbol, state in raw_states.items()
    }


@router.get("/states/{symbol}", response_model=RegimeState)
async def get_regime_for_symbol(
    symbol: str, current_user: User = Depends(get_current_user)
):
    state = regime_monitor.get(symbol.upper())
    if not state:
        return {"error": f"No regime data for {symbol}. Feed price data first."}
    data = state.to_dict() if hasattr(state, "to_dict") else state
    return RegimeState(**data)


@router.get("/correlation", response_model=CorrelationMatrixResponse)
async def get_correlation_matrix(current_user: User = Depends(get_current_user)):
    """Live cross‑strategy correlation matrix."""
    return CorrelationMatrixResponse(
        matrix=correlation_monitor.matrix_as_list(),
        reduced_strategies=list(correlation_monitor._reduced),
        recent_alerts=correlation_monitor.recent_alerts(10),
    )


@router.get("/correlation/alerts", response_model=AlertsResponse)
async def get_correlation_alerts(current_user: User = Depends(get_current_user)):
    """Retrieve recent correlation alerts."""
    return AlertsResponse(alerts=correlation_monitor.recent_alerts(50))