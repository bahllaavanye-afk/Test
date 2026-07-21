"""API endpoints for managing ML experiment tracking.

Provides functionality to list experiments, trigger new training runs,
list available training configurations, and retrieve detailed information
about a specific experiment.
"""

import asyncio
import logging
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.api.deps import get_current_user
from app.models.experiment import Experiment
from app.models.user import User
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parents[4] / "experiments" / "configs"

router = APIRouter(prefix="/experiments", tags=["experiments"])


class ExperimentOut(BaseModel):
    """Schema for exposing experiment summary information via the API."""

    id: str
    name: str
    status: str
    val_accuracy: float | None
    val_sharpe: float | None
    test_sharpe: float | None
    started_at: datetime | None
    completed_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


@router.get("/", response_model=list[ExperimentOut])
async def list_experiments(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[ExperimentOut]:
    """Return the most recent experiments.

    Retrieves up to 50 experiments ordered by start time (newest first).

    Args:
        db: Async SQLAlchemy session.
        current_user: Authenticated user.

    Returns:
        A list of ``ExperimentOut`` instances.
    """
    result = await db.execute(
        select(Experiment).order_by(Experiment.started_at.desc()).limit(50)
    )
    return result.scalars().all()


class TrainRequest(BaseModel):
    """Request payload for queuing a new training experiment."""

    config_name: str  # e.g. "lstm_btc_1h"


async def _run_experiment_async(config_name: str, experiment_id: str) -> None:
    """Background task that executes the experiment script.

    The script is launched as a subprocess using the current Python interpreter.
    All output is silenced; any exception is logged.

    Args:
        config_name: Name of the configuration (without ``.yaml`` suffix).
        experiment_id: Unique identifier for the experiment.
    """
    import subprocess
    import sys

    script = Path(__file__).parents[4] / "experiments" / "run_experiment.py"
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--experiment-id",
            experiment_id,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await proc.wait()
    except Exception as exc:  # pragma: no cover
        logger.error("Experiment %s failed: %s", experiment_id, exc)


@router.post("/train")
async def trigger_training(
    body: TrainRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Queue a training run from an experiment config YAML.

    The request returns immediately with the generated ``experiment_id`` and a
    status of ``queued``. The actual training is performed asynchronously in the
    background.

    Args:
        body: Payload containing the ``config_name`` to use.
        db: Async SQLAlchemy session.
        current_user: Authenticated user.

    Returns:
        A dictionary with ``experiment_id``, ``status`` and ``config_name``.
    """
    config_name = body.config_name.removesuffix(".yaml")

    # Validate config exists
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        available = sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
        raise HTTPException(
            404,
            f"Config '{config_name}' not found. Available: {available[:10]}{'...' if len(available) > 10 else ''}",
        )

    experiment_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    exp = Experiment(
        id=experiment_id,
        name=f"{config_name}-{now.strftime('%Y%m%d%H%M%S')}",
        config={"config_name": config_name},
        status="queued",
        started_at=now,
        created_at=now,
    )
    db.add(exp)
    await db.commit()

    # Launch background training task (fire-and-forget)
    asyncio.create_task(_run_experiment_async(config_name, experiment_id))

    return {
        "experiment_id": experiment_id,
        "status": "queued",
        "config_name": config_name,
    }


@router.get("/train/configs")
async def list_train_configs(
    current_user: User = Depends(get_current_user),
) -> Dict[str, List[str]]:
    """List available training configuration names.

    Args:
        current_user: Authenticated user.

    Returns:
        A dictionary with a single key ``configs`` containing a list of config names.
    """
    if not CONFIGS_DIR.exists():
        return {"configs": []}
    configs = sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
    return {"configs": configs}


@router.get("/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Retrieve detailed information for a specific experiment.

    Args:
        experiment_id: Unique identifier of the experiment.
        db: Async SQLAlchemy session.
        current_user: Authenticated user.

    Returns:
        A dictionary with experiment metadata, metrics, and timestamps.

    Raises:
        HTTPException: If the experiment does not exist.
    """
    result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return {
        "id": exp.id,
        "name": exp.name,
        "config": exp.config,
        "status": exp.status,
        "val_accuracy": exp.val_accuracy,
        "val_sharpe": exp.val_sharpe,
        "test_sharpe": exp.test_sharpe,
        "metrics_history": exp.metrics_history,
        "started_at": exp.started_at,
        "completed_at": exp.completed_at,
    }