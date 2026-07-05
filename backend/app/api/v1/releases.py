"""
ML Model Release Management API.

Endpoints for the full model serving lifecycle:
    registered → shadow → challenger → champion → archived

Also handles A/B test setup, metrics comparison, and inference log access.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models.inference_log import InferenceLog
from app.models.model_release import ModelRelease
from app.models.user import User
from app.utils.logging import logger

router = APIRouter(prefix="/releases", tags=["releases"])


# ─── Schemas ─────────────────────────────────────────────────────────────────


class ReleaseCreate(BaseModel):
    model_config = ConfigDict(strict=True)

    model_name: str = Field(..., max_length=64)
    version: str = Field(..., max_length=32)
    artifact_path: str = Field(..., max_length=512)
    framework: str = Field("pytorch", max_length=32)
    n_features: int | None = None
    seq_len: int | None = None
    model_params: dict = Field(default_factory=dict)
    training_config: dict = Field(default_factory=dict)
    train_metrics: dict = Field(default_factory=dict)
    notes: str | None = None


class ReleaseUpdate(BaseModel):
    notes: str | None = None
    train_metrics: dict | None = None
    live_metrics: dict | None = None


class ChallengeRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    traffic_pct: float = Field(..., ge=1.0, le=50.0)


class OutcomeRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    symbol: str
    actual_return: float
    # ISO-8601 timestamp of the inference to update; if omitted, updates most recent
    ts: str | None = None


class ReleaseOut(BaseModel):
    id: str
    model_name: str
    version: str
    artifact_path: str
    framework: str
    n_features: int | None
    seq_len: int | None
    model_params: dict
    training_config: dict
    train_metrics: dict
    live_metrics: dict
    status: str
    traffic_pct: float
    notes: str | None
    promoted_at: datetime | None
    archived_at: datetime | None
    created_by: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InferenceLogOut(BaseModel):
    id: str
    release_id: str
    model_name: str
    version: str
    symbol: str
    ts: datetime
    prediction: float
    signal: str
    confidence: float
    latency_ms: float
    ab_group: str
    actual_return: float | None
    is_correct: bool | None

    model_config = ConfigDict(from_attributes=True)


class ABStats(BaseModel):
    n_predictions: int
    avg_confidence: float | None
    accuracy: float | None  # fraction of is_correct=True predictions
    avg_latency_ms: float | None


class ABTestMetrics(BaseModel):
    champion: ReleaseOut
    challenger: ReleaseOut
    champion_stats: ABStats
    challenger_stats: ABStats
    recommendation: str  # "promote_challenger" | "keep_champion" | "insufficient_data"
    min_samples_needed: int
    samples_collected: int


# ─── Helpers ─────────────────────────────────────────────────────────────────

_MIN_SAMPLES = 30  # minimum predictions before making a recommendation


def _f(val: Any) -> float | None:
    """Safely cast Decimal → float."""
    return float(val) if val is not None else None


def _release_out(r: ModelRelease) -> ReleaseOut:
    """Convert a ModelRelease ORM instance into a ReleaseOut pydantic model."""
    return ReleaseOut(
        id=r.id,
        model_name=r.model_name,
        version=r.version,
        artifact_path=r.artifact_path,
        framework=r.framework,
        n_features=r.n_features,
        seq_len=r.seq_len,
        model_params=r.model_params or {},
        training_config=r.training_config or {},
        train_metrics=r.train_metrics or {},
        live_metrics=r.live_metrics or {},
        status=r.status,
        traffic_pct=float(r.traffic_pct),
        notes=r.notes,
        promoted_at=r.promoted_at,
        archived_at=r.archived_at,
        created_by=r.created_by,
        created_at=r.created_at,
        updated_at=r.updated_at,
    )


async def _get_release(release_id: str, db: AsyncSession) -> ModelRelease:
    """Fetch a ModelRelease by ID or raise 404."""
    result = await db.execute(select(ModelRelease).where(ModelRelease.id == release_id))
    release = result.scalar_one_or_none()
    if release is None:
        raise HTTPException(404, f"Release '{release_id}' not found")
    return release


async def _compute_ab_stats(release_id: str, db: AsyncSession) -> ABStats:
    """Aggregate inference log metrics for a single release."""
    result = await db.execute(
        select(
            func.count(InferenceLog.id).label("n"),
            func.avg(InferenceLog.confidence).label("avg_conf"),
            func.avg(InferenceLog.latency_ms).label("avg_lat"),
            func.sum(
                func.cast(InferenceLog.is_correct, type_=func.count(InferenceLog.id).type)
            ).label("correct"),
        ).where(InferenceLog.release_id == release_id)
    )
    row = result.one()
    n = int(row.n or 0)
    accuracy: float | None = None
    if n > 0 and row.correct is not None:
        accuracy = float(row.correct) / n

    return ABStats(
        n_predictions=n,
        avg_confidence=_f(row.avg_conf),
        accuracy=accuracy,
        avg_latency_ms=_f(row.avg_lat),
    )


async def _build_ab_metrics(
    champion: ModelRelease,
    challenger: ModelRelease,
    db: AsyncSession,
) -> ABTestMetrics:
    """Construct AB test comparison metrics for two releases."""
    ch_stats = await _compute_ab_stats(champion.id, db)
    cl_stats = await _compute_ab_stats(challenger.id, db)

    samples = min(ch_stats.n_predictions, cl_stats.n_predictions)
    recommendation = "insufficient_data"
    if samples >= _MIN_SAMPLES:
        ch_acc = ch_stats.accuracy or 0.0
        cl_acc = cl_stats.accuracy or 0.0
        recommendation = "promote_challenger" if cl_acc > ch_acc else "keep_champion"

    return ABTestMetrics(
        champion=_release_out(champion),
        challenger=_release_out(challenger),
        champion_stats=ch_stats,
        challenger_stats=cl_stats,
        recommendation=recommendation,
        min_samples_needed=_MIN_SAMPLES,
        samples_collected=samples,
    )


def _invalidate_router(model_name: str) -> None:
    """Purge the A/B router snapshot for *model_name* after any status change."""
    try:
        from app.ml.serving.ab_router import get_ab_router

        get_ab_router().invalidate(model_name)
    except Exception:
        # Router may not be initialised yet; safe to ignore.
        pass


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/", response_model=list[ReleaseOut])
async def list_releases(
    model_name: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100, le=500),
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[ReleaseOut]:
    """List all model releases, newest first. Filter by model_name or status."""
    stmt = select(ModelRelease).order_by(ModelRelease.created_at.desc()).limit(limit)
    if model_name:
        stmt = stmt.where(ModelRelease.model_name == model_name)
    if status:
        stmt = stmt.where(ModelRelease.status == status)
    result = await db.execute(stmt)
    return [_release_out(r) for r in result.scalars().all()]


@router.post("/", response_model=ReleaseOut, status_code=201)
async def register_release(
    body: ReleaseCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ReleaseOut:
    """Register a newly trained model artifact for serving."""
    release = ModelRelease(
        id=str(uuid.uuid4()),
        model_name=body.model_name,
        version=body.version,
        artifact_path=body.artifact_path,
        framework=body.framework,
        n_features=body.n_features,
        seq_len=body.seq_len,
        model_params=body.model_params,
        training_config=body.training_config,
        train_metrics=body.train_metrics,
        notes=body.notes,
        status="registered",
        traffic_pct=0.0,
        created_by=getattr(user, "email", str(user)),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(release)
    await db.commit()
    await db.refresh(release)
    return _release_out(release)


@router.get("/{release_id}", response_model=ReleaseOut)
async def get_release(
    release_id: str,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ReleaseOut:
    """Retrieve a single model release by its ID."""
    release = await _get_release(release_id, db)
    return _release_out(release)


@router.patch("/{release_id}", response_model=ReleaseOut)
async def update_release(
    release_id: str,
    body: ReleaseUpdate,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ReleaseOut:
    """Update mutable fields of a model release."""
    release = await _get_release(release_id, db)

    if body.notes is not None:
        release.notes = body.notes
    if body.train_metrics is not None:
        release.train_metrics = body.train_metrics
    if body.live_metrics is not None:
        release.live_metrics = body.live_metrics

    release.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(release)
    return _release_out(release)


@router.post("/{release_id}/challenge", response_model=ReleaseOut)
async def challenge_release(
    release_id: str,
    body: ChallengeRequest,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ReleaseOut:
    """Promote a registered release to challenger status and allocate traffic."""
    release = await _get_release(release_id, db)

    if release.status != "registered":
        raise HTTPException(400, f"Release '{release_id}' is not in a registerable state")

    release.status = "challenger"
    release.traffic_pct = body.traffic_pct
    release.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(release)

    _invalidate_router(release.model_name)
    return _release_out(release)


@router.post("/{release_id}/outcome", response_model=ReleaseOut)
async def record_outcome(
    release_id: str,
    body: OutcomeRequest,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ReleaseOut:
    """Record the actual return for a specific inference, updating correctness."""
    release = await _get_release(release_id, db)

    # Locate the inference log entry
    stmt = select(InferenceLog).where(
        InferenceLog.release_id == release_id,
        InferenceLog.symbol == body.symbol,
    )
    if body.ts:
        stmt = stmt.where(InferenceLog.ts == datetime.fromisoformat(body.ts))
    else:
        stmt = stmt.order_by(InferenceLog.ts.desc()).limit(1)

    result = await db.execute(stmt)
    log = result.scalar_one_or_none()
    if log is None:
        raise HTTPException(404, "Inference log entry not found for given parameters")

    log.actual_return = body.actual_return
    # Define correctness as sign(prediction) == sign(actual_return)
    log.is_correct = (log.prediction * body.actual_return) >= 0

    # Update live metrics on the release (simple aggregation)
    release.live_metrics = release.live_metrics or {}
    release.live_metrics.setdefault("last_update", datetime.now(timezone.utc).isoformat())
    release.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(release)
    return _release_out(release)


@router.get("/{release_id}/inferences", response_model=list[InferenceLogOut])
async def list_inference_logs(
    release_id: str,
    limit: int = Query(100, le=500),
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[InferenceLogOut]:
    """Return recent inference logs for a given release."""
    stmt = (
        select(InferenceLog)
        .where(InferenceLog.release_id == release_id)
        .order_by(InferenceLog.ts.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [InferenceLogOut.from_orm(log) for log in result.scalars().all()]


@router.get("/ab_test", response_model=ABTestMetrics)
async def ab_test_metrics(
    champion_id: str,
    challenger_id: str,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ABTestMetrics:
    """Compute AB test comparison metrics between two releases."""
    champion = await _get_release(champion_id, db)
    challenger = await _get_release(challenger_id, db)

    if champion.model_name != challenger.model_name:
        raise HTTPException(
            400,
            "Champion and challenger must belong to the same model_name for AB testing",
        )

    return await _build_ab_metrics(champion, challenger, db)