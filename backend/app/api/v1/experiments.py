"""ML experiment tracking endpoints."""
import asyncio
import logging
import uuid
from pathlib import Path
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.experiment import Experiment
from app.models.user import User
from pydantic import BaseModel, ConfigDict
from datetime import datetime, timezone

# Constants
CONFIGS_DIR = Path(__file__).parents[4] / "experiments" / "configs"
RUN_EXPERIMENT_SCRIPT = Path(__file__).parents[4] / "experiments" / "run_experiment.py"
CONFIG_SUFFIX = ".yaml"
LIST_EXPERIMENTS_LIMIT = 50
STATUS_QUEUED = "queued"
EXPERIMENT_NOT_FOUND_MSG = "Experiment not found"
MAX_DISPLAYED_CONFIGS = 10
CONFIG_NOT_FOUND_TEMPLATE = "Config '{config}' not found. Available: {available}"
EXPERIMENT_RESPONSE_FIELDS = {
    "experiment_id": "experiment_id",
    "status": "status",
    "config_name": "config_name",
}
METRICS_RESPONSE_FIELDS = {
    "id": "id",
    "name": "name",
    "config": "config",
    "status": "status",
    "val_accuracy": "val_accuracy",
    "val_sharpe": "val_sharpe",
    "test_sharpe": "test_sharpe",
    "metrics_history": "metrics_history",
    "started_at": "started_at",
    "completed_at": "completed_at",
}

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])


class ExperimentOut(BaseModel):
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
):
    result = await db.execute(
        select(Experiment).order_by(Experiment.started_at.desc()).limit(LIST_EXPERIMENTS_LIMIT)
    )
    return result.scalars().all()


class TrainRequest(BaseModel):
    config_name: str  # e.g. "lstm_btc_1h"


async def _run_experiment_async(config_name: str, experiment_id: str) -> None:
    """Background task: run the experiment script for the given config."""
    import subprocess
    import sys

    config_path = CONFIGS_DIR / f"{config_name}{CONFIG_SUFFIX}"
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(RUN_EXPERIMENT_SCRIPT),
            "--config",
            str(config_path),
            "--experiment-id",
            experiment_id,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await proc.wait()
    except Exception as exc:
        logger.error("Experiment %s failed: %s", experiment_id, exc)


@router.post("/train")
async def trigger_training(
    body: TrainRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Queue a training run from an experiment config YAML.

    Returns immediately with experiment_id and status='queued'.
    The training runs as a background asyncio task.
    """
    config_name = body.config_name.removesuffix(CONFIG_SUFFIX)

    # Validate config exists
    config_path = CONFIGS_DIR / f"{config_name}{CONFIG_SUFFIX}"
    if not config_path.exists():
        available = sorted(p.stem for p in CONFIGS_DIR.glob(f"*{CONFIG_SUFFIX}"))
        displayed = available[:MAX_DISPLAYED_CONFIGS]
        suffix = "..." if len(available) > MAX_DISPLAYED_CONFIGS else ""
        raise HTTPException(
            404,
            CONFIG_NOT_FOUND_TEMPLATE.format(
                config=config_name,
                available=f"{displayed}{suffix}"
            ),
        )

    experiment_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    exp = Experiment(
        id=experiment_id,
        name=f"{config_name}-{now.strftime('%Y%m%d%H%M%S')}",
        config={"config_name": config_name},
        status=STATUS_QUEUED,
        started_at=now,
        created_at=now,
    )
    db.add(exp)
    await db.commit()

    # Launch background training task (fire-and-forget)
    asyncio.create_task(_run_experiment_async(config_name, experiment_id))

    return {
        EXPERIMENT_RESPONSE_FIELDS["experiment_id"]: experiment_id,
        EXPERIMENT_RESPONSE_FIELDS["status"]: STATUS_QUEUED,
        EXPERIMENT_RESPONSE_FIELDS["config_name"]: config_name,
    }


@router.get("/train/configs")
async def list_train_configs(
    current_user: User = Depends(get_current_user),
):
    """List available training config names."""
    if not CONFIGS_DIR.exists():
        return {"configs": []}
    configs = sorted(p.stem for p in CONFIGS_DIR.glob(f"*{CONFIG_SUFFIX}"))
    return {"configs": configs}


@router.get("/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(404, EXPERIMENT_NOT_FOUND_MSG)
    return {
        METRICS_RESPONSE_FIELDS["id"]: exp.id,
        METRICS_RESPONSE_FIELDS["name"]: exp.name,
        METRICS_RESPONSE_FIELDS["config"]: exp.config,
        METRICS_RESPONSE_FIELDS["status"]: exp.status,
        METRICS_RESPONSE_FIELDS["val_accuracy"]: exp.val_accuracy,
        METRICS_RESPONSE_FIELDS["val_sharpe"]: exp.val_sharpe,
        METRICS_RESPONSE_FIELDS["test_sharpe"]: exp.test_sharpe,
        METRICS_RESPONSE_FIELDS["metrics_history"]: exp.metrics_history,
        METRICS_RESPONSE_FIELDS["started_at"]: exp.started_at,
        METRICS_RESPONSE_FIELDS["completed_at"]: exp.completed_at,
    }