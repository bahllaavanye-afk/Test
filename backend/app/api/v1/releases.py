"""
ML Model Release Management API.

Endpoints for the full model serving lifecycle:
    registered → shadow → challenger → champion → archived

Also handles A/B test setup, metrics comparison, and inference log access.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
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
    accuracy: float | None          # fraction of is_correct=True predictions
    avg_latency_ms: float | None


class ABTestMetrics(BaseModel):
    champion: ReleaseOut
    challenger: ReleaseOut
    champion_stats: ABStats
    challenger_stats: ABStats
    recommendation: str             # "promote_challenger" | "keep_champion" | "insufficient_data"
    min_samples_needed: int
    samples_collected: int


# ─── Helpers ─────────────────────────────────────────────────────────────────

_MIN_SAMPLES = 30  # minimum predictions before making a recommendation


def _f(val: Any) -> float | None:
    """Safely cast Decimal → float."""
    return float(val) if val is not None else None


def _release_out(r: ModelRelease) -> ReleaseOut:
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
    result = await db.execute(
        select(ModelRelease).where(ModelRelease.id == release_id)
    )
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
        pass  # router may not be initialised yet


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
    q = select(ModelRelease).order_by(ModelRelease.created_at.desc()).limit(limit)
    if model_name:
        q = q.where(ModelRelease.model_name == model_name)
    if status:
        q = q.where(ModelRelease.status == status)
    result = await db.execute(q)
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
        live_metrics={},
        status="registered",
        traffic_pct=0.0,
        notes=body.notes,
        created_by=getattr(user, "email", str(user.id)),
    )
    db.add(release)
    await db.commit()
    await db.refresh(release)
    return _release_out(release)


@router.get("/champion/{model_name}", response_model=ReleaseOut)
async def get_champion(
    model_name: str,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ReleaseOut:
    """Return the current champion for *model_name*."""
    result = await db.execute(
        select(ModelRelease).where(
            ModelRelease.model_name == model_name,
            ModelRelease.status == "champion",
        )
    )
    champion = result.scalar_one_or_none()
    if champion is None:
        raise HTTPException(404, f"No champion found for model '{model_name}'")
    return _release_out(champion)


@router.get("/ab-tests/active", response_model=list[ABTestMetrics])
async def list_active_ab_tests(
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[ABTestMetrics]:
    """Return metrics for all currently running A/B tests."""
    # Find all challengers
    challengers_q = await db.execute(
        select(ModelRelease).where(ModelRelease.status == "challenger")
    )
    challengers = challengers_q.scalars().all()

    results = []
    for challenger in challengers:
        # Find its champion
        champ_q = await db.execute(
            select(ModelRelease).where(
                ModelRelease.model_name == challenger.model_name,
                ModelRelease.status == "champion",
            )
        )
        champion = champ_q.scalar_one_or_none()
        if champion is None:
            continue
        metrics = await _build_ab_metrics(champion, challenger, db)
        results.append(metrics)

    return results


@router.get("/{release_id}", response_model=ReleaseOut)
async def get_release(
    release_id: str,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ReleaseOut:
    return _release_out(await _get_release(release_id, db))


@router.patch("/{release_id}", response_model=ReleaseOut)
async def update_release(
    release_id: str,
    body: ReleaseUpdate,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ReleaseOut:
    """Update mutable fields: notes, train_metrics, live_metrics."""
    release = await _get_release(release_id, db)
    if body.notes is not None:
        release.notes = body.notes
    if body.train_metrics is not None:
        release.train_metrics = body.train_metrics
    if body.live_metrics is not None:
        release.live_metrics = body.live_metrics
    await db.commit()
    await db.refresh(release)
    return _release_out(release)


@router.post("/{release_id}/shadow", response_model=ReleaseOut)
async def set_shadow(
    release_id: str,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ReleaseOut:
    """
    Move a release to shadow status (traffic_pct=0).

    Shadow releases receive no live traffic but their predictions are logged
    for offline comparison against the champion.
    """
    release = await _get_release(release_id, db)
    if release.status not in ("registered", "challenger"):
        raise HTTPException(
            400,
            f"Only registered or challenger releases can become shadow "
            f"(current status: {release.status})",
        )
    release.status = "shadow"
    release.traffic_pct = 0.0
    await db.commit()
    await db.refresh(release)
    _invalidate_router(release.model_name)
    return _release_out(release)


@router.post("/{release_id}/challenge", response_model=ReleaseOut)
async def start_ab_test(
    release_id: str,
    body: ChallengeRequest,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ReleaseOut:
    """
    Start an A/B test by making this release a challenger.

    Rules:
    - Only one challenger per model_name is allowed.
    - A champion must exist for the model_name.
    - traffic_pct must be 1–50 % (challenger never gets majority by design).
    """
    release = await _get_release(release_id, db)

    if release.status == "champion":
        raise HTTPException(400, "Cannot challenge with the current champion")
    if release.status == "archived":
        raise HTTPException(400, "Cannot challenge with an archived release")

    # Ensure a champion exists
    champ_q = await db.execute(
        select(ModelRelease).where(
            ModelRelease.model_name == release.model_name,
            ModelRelease.status == "champion",
        )
    )
    if champ_q.scalar_one_or_none() is None:
        raise HTTPException(
            400,
            f"No champion exists for model '{release.model_name}'. "
            "Promote a release to champion first.",
        )

    # Ensure no other challenger exists
    existing_q = await db.execute(
        select(ModelRelease).where(
            ModelRelease.model_name == release.model_name,
            ModelRelease.status == "challenger",
            ModelRelease.id != release_id,
        )
    )
    if existing_q.scalar_one_or_none() is not None:
        raise HTTPException(
            400,
            f"Already have an active challenger for '{release.model_name}'. "
            "Archive or promote it before starting a new test.",
        )

    release.status = "challenger"
    release.traffic_pct = body.traffic_pct
    await db.commit()
    await db.refresh(release)
    _invalidate_router(release.model_name)
    return _release_out(release)


@router.post("/{release_id}/promote", response_model=ReleaseOut)
async def promote_to_champion(
    release_id: str,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ReleaseOut:
    """Promote-to-live is permanently disabled — platform is paper-only."""
    raise HTTPException(
        403,
        "Promote-to-live is disabled: QuantEdge runs in paper mode only. "
        "Remove PAPER_ONLY_POLICY=true from config to re-enable (not recommended).",
    )
    release = await _get_release(release_id, db)

    if release.status not in ("challenger", "shadow", "registered"):
        raise HTTPException(
            400,
            f"Release must be in challenger, shadow, or registered status to promote "
            f"(current: {release.status})",
        )

    # Archive the existing champion (if any)
    old_champ_q = await db.execute(
        select(ModelRelease).where(
            ModelRelease.model_name == release.model_name,
            ModelRelease.status == "champion",
        )
    )
    old_champion = old_champ_q.scalar_one_or_none()
    if old_champion and old_champion.id != release_id:
        old_champion.status = "archived"
        old_champion.archived_at = datetime.now(UTC)
        old_champion.traffic_pct = 0.0

    now = datetime.now(UTC)
    release.status = "champion"
    release.traffic_pct = 100.0
    release.promoted_at = now

    await db.commit()
    await db.refresh(release)
    _invalidate_router(release.model_name)

    # Evict from serving cache if loaded
    try:
        from app.ml.serving.serve import get_serving_layer
        get_serving_layer().invalidate_model(release_id)
        if old_champion:
            get_serving_layer().invalidate_model(old_champion.id)
    except Exception as exc:
        logger.debug("serving cache invalidation failed", release_id=release_id, error=str(exc))

    return _release_out(release)


@router.post("/{release_id}/archive", response_model=ReleaseOut)
async def archive_release(
    release_id: str,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ReleaseOut:
    """
    Archive a release (stops serving traffic).

    Champions cannot be archived — promote another release first.
    """
    release = await _get_release(release_id, db)

    if release.status == "champion":
        raise HTTPException(
            400,
            "Cannot archive the champion — promote another release first.",
        )
    if release.status == "archived":
        raise HTTPException(400, "Release is already archived.")

    release.status = "archived"
    release.traffic_pct = 0.0
    release.archived_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(release)
    _invalidate_router(release.model_name)
    return _release_out(release)


@router.get("/{release_id}/metrics", response_model=ABTestMetrics)
async def get_ab_metrics(
    release_id: str,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ABTestMetrics:
    """
    Return A/B test comparison metrics for this release vs the champion.

    Works for challengers (mid-test) and shadow releases (offline analysis).
    """
    release = await _get_release(release_id, db)

    champ_q = await db.execute(
        select(ModelRelease).where(
            ModelRelease.model_name == release.model_name,
            ModelRelease.status == "champion",
        )
    )
    champion = champ_q.scalar_one_or_none()
    if champion is None:
        raise HTTPException(
            404,
            f"No champion found for model '{release.model_name}' to compare against",
        )
    if champion.id == release.id:
        raise HTTPException(400, "This release is the champion — compare a challenger against it")

    return await _build_ab_metrics(champion, release, db)


@router.get("/{release_id}/inferences", response_model=list[InferenceLogOut])
async def get_inference_logs(
    release_id: str,
    limit: int = Query(100, le=1000),
    symbol: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[InferenceLogOut]:
    """Return recent inference logs for this release, newest first."""
    await _get_release(release_id, db)  # 404 guard

    q = (
        select(InferenceLog)
        .where(InferenceLog.release_id == release_id)
        .order_by(InferenceLog.ts.desc())
        .limit(limit)
    )
    if symbol:
        q = q.where(InferenceLog.symbol == symbol)

    result = await db.execute(q)
    return list(result.scalars().all())


@router.post("/{release_id}/record-outcome")
async def record_outcome(
    release_id: str,
    body: OutcomeRequest,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> dict:
    """
    Record the actual return outcome for a recent inference.

    Fills InferenceLog.actual_return and sets is_correct based on whether
    the predicted direction (buy=positive, sell=negative) matches the actual return.
    Updates the most recent unresolved inference for (release_id, symbol) unless
    a specific timestamp is provided.
    """
    await _get_release(release_id, db)  # 404 guard

    q = (
        select(InferenceLog)
        .where(
            InferenceLog.release_id == release_id,
            InferenceLog.symbol == body.symbol,
            InferenceLog.actual_return.is_(None),
        )
        .order_by(InferenceLog.ts.desc())
        .limit(1)
    )
    if body.ts:
        try:
            ts_dt = datetime.fromisoformat(body.ts)
            q = select(InferenceLog).where(
                InferenceLog.release_id == release_id,
                InferenceLog.symbol == body.symbol,
                InferenceLog.ts == ts_dt,
            ).limit(1)
        except ValueError:
            raise HTTPException(400, "Invalid timestamp format — use ISO-8601")

    result = await db.execute(q)
    log_entry = result.scalar_one_or_none()
    if log_entry is None:
        raise HTTPException(404, "No unresolved inference found for this release/symbol")

    log_entry.actual_return = body.actual_return
    log_entry.is_correct = (
        (log_entry.signal == "buy" and body.actual_return > 0) or
        (log_entry.signal == "sell" and body.actual_return < 0)
    )
    await db.commit()
    return {"updated": log_entry.id, "is_correct": log_entry.is_correct}
