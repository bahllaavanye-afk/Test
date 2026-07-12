"""Market regime and cross‑strategy correlation endpoints."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

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
        ge=0.0,
        le=1.0,
        description="Average confidence (0‑1) of the regime classification.",
        example=0.842,
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Timestamp of the most recent regime update (ISO 8601).",
        example="2024-11-05T14:32:10Z",
    )
    symbol_count: int = Field(
        ...,
        ge=0,
        description="Number of symbols contributing to the aggregated regime.",
        example=124,
    )

    @validator("regime")
    def validate_regime(cls, v: str) -> str:
        allowed = {"bull", "bear", "sideways", "unknown"}
        if v not in allowed:
            raise ValueError(f"regime must be one of {allowed}")
        return v


class SymbolRegimeResponse(BaseModel):
    """Regime classification for a single symbol."""

    symbol: str = Field(..., description="Ticker symbol.", example="AAPL")
    regime: str = Field(..., description="Regime label for the symbol.", example="sideways")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of the classification.", example=0.71)
    updated_at: Optional[datetime] = Field(
        None,
        description="Timestamp of the latest regime update for the symbol.",
        example="2024-11-05T14:30:00Z",
    )

    @validator("regime")
    def validate_regime(cls, v: str) -> str:
        allowed = {"bull", "bear", "sideways", "unknown"}
        if v not in allowed:
            raise ValueError(f"regime must be one of {allowed}")
        return v


class CorrelationResponse(BaseModel):
    """Live cross‑strategy correlation matrix."""

    matrix: List[List[float]] = Field(
        ...,
        description="Correlation coefficients between strategies (range -1 to 1).",
        example=[[1.0, 0.23, -0.11], [0.23, 1.0, 0.05], [-0.11, 0.05, 1.0]],
    )
    reduced_strategies: List[str] = Field(
        ...,
        description="Names of strategies retained after dimensionality reduction.",
        example=["mean_rev_20_1.5", "trend_50_2.0"],
    )
    recent_alerts: List[Dict[str, Any]] = Field(
        ...,
        description="Most recent correlation alerts.",
        example=[{"strategy": "mean_rev_20_1.5", "alert": "high correlation", "timestamp": "2024-11-05T14:20:00Z"}],
    )


class AlertsResponse(BaseModel):
    """Correlation alerts list."""

    alerts: List[Dict[str, Any]] = Field(
        ...,
        description="Correlation alerts ordered by recency (most recent first).",
        example=[{"strategy": "trend_30_1.2", "alert": "spike", "timestamp": "2024-11-05T14:05:00Z"}],
    )


@router.get("/current", response_model=CurrentRegimeResponse)
async def get_current_regime(current_user: User = Depends(get_current_user)):
    """Overall market regime — aggregated across all tracked symbols.

    Returns the most common regime (bull/bear/sideways) and average confidence.
    Falls back to safe defaults when no data is available.
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
    latest_updated: Optional[datetime] = None

    for sym_state in states.values():
        raw = sym_state.get("regime", "unknown")
        label = _label_map.get(raw, "unknown")
        label_counts[label] += 1
        confidences.append(sym_state.get("confidence", 0.0))
        updated_str = sym_state.get("updated_at")
        if updated_str:
            try:
                updated_dt = datetime.fromisoformat(updated_str.rstrip("Z"))
                if latest_updated is None or updated_dt > latest_updated:
                    latest_updated = updated_dt
            except ValueError:
                # Keep the original string if parsing fails
                latest_updated = None

    overall_regime = label_counts.most_common(1)[0][0]
    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    return CurrentRegimeResponse(
        regime=overall_regime,
        confidence=avg_confidence,
        updated_at=latest_updated,
        symbol_count=len(states),
    )


@router.get("/states")
async def get_regime_states(current_user: User = Depends(get_current_user)):
    """Current regime classification for all tracked symbols."""
    return regime_monitor.all_states()


@router.get("/states/{symbol}", response_model=SymbolRegimeResponse)
async def get_regime_for_symbol(symbol: str, current_user: User = Depends(get_current_user)):
    state = regime_monitor.get(symbol.upper())
    if not state:
        return {"error": f"No regime data for {symbol}. Feed price data first."}
    data = state.to_dict()
    return SymbolRegimeResponse(
        symbol=symbol.upper(),
        regime=data.get("regime", "unknown"),
        confidence=data.get("confidence", 0.0),
        updated_at=data.get("updated_at"),
    )


@router.get("/correlation", response_model=CorrelationResponse)
async def get_correlation_matrix(current_user: User = Depends(get_current_user)):
    """Live cross‑strategy correlation matrix."""
    return CorrelationResponse(
        matrix=correlation_monitor.matrix_as_list(),
        reduced_strategies=list(correlation_monitor._reduced),
        recent_alerts=correlation_monitor.recent_alerts(10),
    )


@router.get("/correlation/alerts", response_model=AlertsResponse)
async def get_correlation_alerts(current_user: User = Depends(get_current_user)):
    """Retrieve recent correlation alerts."""
    return AlertsResponse(alerts=correlation_monitor.recent_alerts(50))