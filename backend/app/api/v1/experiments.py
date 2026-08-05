"""ML experiment tracking endpoints."""
import asyncio
import logging
import uuid
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.api.deps import get_current_user
from app.models.experiment import Experiment
from app.models.user import User
from pydantic import BaseModel, ConfigDict

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
        select(Experiment).order_by(Experiment.started_at.desc()).limit(MAX_EXPERIMENTS)
    )
    return result.scalars().all()


class TrainRequest(BaseModel):
    config_name: str  # e.g. "lstm_btc_1h"


async def _run_experiment_async(config_name: str, experiment_id: str) -> None:
    """Background task: run the experiment script for the given config."""
    import subprocess
    import sys

    script = Path(__file__).parents[4] / "experiments" / "run_experiment.py"
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    start_time = datetime.now(timezone.utc)

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
    except Exception as exc:
        logger.error(
            "Experiment %s failed: %s",
            experiment_id,
            exc,
            extra={"experiment_id": experiment_id, "config_name": config_name},
        )
        return

    duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info(
        "Experiment completed",
        extra={
            "experiment_id": experiment_id,
            "config_name": config_name,
            "duration_seconds": duration_seconds,
            # Placeholders for future metrics
            "signal_count": None,
            "pnl": None,
        },
    )


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
    config_name = body.config_name.removesuffix(".yaml")

    # Validate config exists
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        available = sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
        display_list = available[:CONFIGS_LIMIT_DISPLAY]
        suffix = "..." if len(available) > CONFIGS_LIMIT_DISPLAY else ""
        message = CONFIG_NOT_FOUND_TEMPLATE.format(
            config_name=config_name,
            available=f"{display_list}{suffix}",
        )
        raise HTTPException(404, message)

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

    logger.info(
        "Experiment queued",
        extra={
            "experiment_id": experiment_id,
            "config_name": config_name,
            "queued_at": now.isoformat(),
        },
    )

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
):
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
):
    result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
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