"""Self-improvement history endpoint."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, validator

from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/improvements", tags=["improvements"])


class ImprovementHistoryItem(BaseModel):
    """A single entry in the self‑improvement history."""

    timestamp: datetime = Field(
        ...,
        description="When the improvement entry was recorded (UTC).",
        example="2023-09-15T14:23:00Z",
    )
    improvement: str = Field(
        ...,
        description="Brief description of the improvement performed.",
        example="Refactored order execution module for latency reduction.",
    )
    result: str = Field(
        ...,
        description="Outcome of the improvement (e.g., success, failure, notes).",
        example="success",
    )

    @validator("timestamp")
    def timestamp_not_in_future(cls, v: datetime) -> datetime:
        """Ensure the timestamp is not set in the future."""
        if v > datetime.utcnow():
            raise ValueError("timestamp cannot be in the future")
        return v


class QualityStatus(str):
    """Allowed status values for the quality endpoint."""

    NOT_RUNNING = "not_running"
    RUNNING = "running"


class CodeQualityResponse(BaseModel):
    """Current status of the code‑quality monitoring loop."""

    status: str = Field(
        ...,
        description="Current state of the quality loop.",
        example=QualityStatus.RUNNING,
    )
    message: Optional[str] = Field(
        None,
        description="Human‑readable message when the loop is not running.",
        example="Code quality loop not started",
    )
    metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Key‑value pairs representing quality metrics (e.g., lint score, test coverage).",
        example={"lint_score": 92.5, "coverage": 0.87},
    )

    @validator("status")
    def validate_status(cls, v: str) -> str:
        allowed = {QualityStatus.NOT_RUNNING, QualityStatus.RUNNING}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v


class BestParamsResponse(BaseModel):
    """Best parameters discovered by the self‑improver."""

    status: str = Field(
        ...,
        description="Indicates whether the self‑improver is active.",
        example="running",
    )
    best_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of the most recent best parameters.",
        example={"learning_rate": 0.001, "batch_size": 64},
    )


@router.get(
    "/history",
    response_model=List[ImprovementHistoryItem],
    summary="Retrieve self‑improvement history",
    description="Returns the chronological list of improvement actions performed by the system.",
)
async def get_history(current_user: User = Depends(get_current_user)):
    from app.main import app

    improver = getattr(app.state, "self_improver", None)
    if improver:
        return improver.get_history()
    return []


@router.get(
    "/quality",
    response_model=CodeQualityResponse,
    summary="Get current code‑quality loop status",
    description="Provides the latest metrics and status of the background code‑quality monitoring loop.",
)
async def get_quality(current_user: User = Depends(get_current_user)):
    from app.main import app

    loop_ref = getattr(app.state, "code_quality_loop", None)
    if loop_ref is None:
        return CodeQualityResponse(
            status=QualityStatus.NOT_RUNNING,
            message="Code quality loop not started",
        )
    latest = loop_ref.latest()
    # Assume `latest` returns a dict with metric data; enrich with status.
    return CodeQualityResponse(status=QualityStatus.RUNNING, metrics=latest)


@router.get(
    "/best_params",
    response_model=BestParamsResponse,
    summary="Retrieve best parameters from self‑improver",
    description="Returns the most recent best parameter set discovered by the self‑improver, if it is running.",
)
async def get_best_params(current_user: User = Depends(get_current_user)):
    from app.main import app

    improver = getattr(app.state, "self_improver", None)
    if improver is None:
        return BestParamsResponse(status="not_running", best_params={})
    return BestParamsResponse(status="running", best_params=getattr(improver, "_best_params", {}))