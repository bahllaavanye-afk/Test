"""ML experiment tracking endpoints."""
import asyncio
import logging
import uuid
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, ConfigDict

from app.database import get_db
from app.api.deps import get_current_user
from app.models.experiment import Experiment
from app.models.user import User

# Constants
MAX_EXPERIMENTS = 50
STATUS_QUEUED = "queued"
EXPERIMENT_NOT_FOUND = "Experiment not found"
CONFIG_NOT_FOUND_TEMPLATE = "Config '{config_name}' not found. Available: {available}"
CONFIGS_LIMIT_DISPLAY = 10
CONFIGS_KEY = "configs"

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
    """Return the most recent experiments limited by ``MAX_EXPERIMENTS``."""
    if MAX_EXPERIMENTS <= 0:
        return []
    try:
        result = await db.execute(
            select(Experiment)
            .order_by(Experiment.started_at.desc())
            .limit(MAX_EXPERIMENTS)
        )
        return result.scalars().all()
    except SQLAlchemyError as exc:
        logger.exception("Failed to list experiments: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to retrieve experiments")


class TrainRequest(BaseModel):
    """Request payload for triggering a training run."""

    config_name: str  # e.g. "lstm_btc_1h"


async def _run_experiment_async(config_name: str, experiment_id: str) -> None:
    """Background task: run the experiment script for the given config."""
    script = Path(__file__).parents[4] / "experiments" / "run_experiment.py"
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    if not config_path.is_file():
        logger.error(
            "Config file not found",
            extra={"config_path": str(config_path), "experiment_id": experiment_id},
        )
        return
    if not script.is_file():
        logger.error(
            "Experiment script not found",
            extra={"script_path": str(script), "experiment_id": experiment_id},
        )
        return
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
        return_code = await proc.wait()
        if return_code != 0:
            logger.error(
                "Experiment subprocess exited with non-zero status",
                extra={
                    "experiment_id": experiment_id,
                    "config_name": config_name,
                    "return_code": return_code,
                },
            )
    except (OSError, asyncio.SubprocessError) as exc:
        logger.exception(
            "Failed to start experiment subprocess",
            extra={"experiment_id": experiment_id, "config_name": config_name},
        )
    except Exception as exc:  # pragma: no cover
        logger.exception(
            "Unexpected error while running experiment",
            extra={"experiment_id": experiment_id, "config_name": config_name},
        )


def _format_config_not_found_message(config_name: str, available: List[str]) -> str:
    """Create a user‑friendly error message when a config file cannot be found."""
    display_list = available[:CONFIGS_LIMIT_DISPLAY]
    suffix = "..." if len(available) > CONFIGS_LIMIT_DISPLAY else ""
    return CONFIG_NOT_FOUND_TEMPLATE.format(
        config_name=config_name,
        available=f"{display_list}{suffix}",
    )


def _validate_config_exists(config_name: str) -> Path:
    """Ensure the YAML config file exists, otherwise raise ``HTTPException``."""
    if not config_name:
        raise HTTPException(status_code=400, detail="Config name must not be empty")
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    if config_path.is_file():
        return config_path
    available = sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
    message = _format_config_not_found_message(config_name, available)
    raise HTTPException(status_code=404, detail=message)


def _create_experiment_record(config_name: str, experiment_id: str, now: datetime) -> Experiment:
    """Construct a new ``Experiment`` ORM instance with initial status."""
    return Experiment(
        id=experiment_id,
        name=f"{config_name}-{now.strftime('%Y%m%d%H%M%S')}",
        config={"config_name": config_name},
        status=STATUS_QUEUED,
        started_at=now,
        created_at=now,
    )


@router.post("/train")
async def trigger_training(
    body: TrainRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """Queue a training run from an experiment config YAML.

    Returns immediately with ``experiment_id`` and ``status='queued'``.
    The training runs as a background asyncio task.
    """
    config_name = body.config_name.removesuffix(".yaml")
    if not config_name:
        raise HTTPException(status_code=400, detail="Config name must not be empty")

    # Validate config existence
    _validate_config_exists(config_name)

    experiment_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Persist experiment record
    exp = _create_experiment_record(config_name, experiment_id, now)
    db.add(exp)
    try:
        await db.commit()
    except SQLAlchemyError as exc:
        logger.exception(
            "Database error while creating experiment record",
            extra={"experiment_id": experiment_id, "config_name": config_name},
        )
        raise HTTPException(status_code=500, detail="Failed to create experiment record")

    # Launch background training task (fire-and-forget)
    asyncio.create_task(_run_experiment_async(config_name, experiment_id))

    return {
        "experiment_id": experiment_id,
        "status": STATUS_QUEUED,
        "config_name": config_name,
    }


@router.get("/train/configs")
async def list_train_configs(
    current_user: User = Depends(get_current_user),
) -> Dict[str, List[str]]:
    """List available training config names."""
    if not CONFIGS_DIR.exists():
        return {CONFIGS_KEY: []}
    configs = sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
    return {CONFIGS_KEY: configs}


@router.get("/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Retrieve detailed information for a specific experiment."""
    if not experiment_id:
        raise HTTPException(status_code=404, detail=EXPERIMENT_NOT_FOUND)
    try:
        result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    except SQLAlchemyError as exc:
        logger.exception(
            "Database error while fetching experiment",
            extra={"experiment_id": experiment_id},
        )
        raise HTTPException(status_code=500, detail="Unable to retrieve experiment")
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(404, EXPERIMENT_NOT_FOUND)
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