"""ML experiment tracking endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.experiment import Experiment
from app.models.user import User
from pydantic import BaseModel, ConfigDict
from datetime import datetime
import json
from pathlib import Path

EXPERIMENTS_RESULTS_DIR = Path(__file__).parents[4] / "experiments" / "results"
MODELS_ARTIFACTS_DIR = Path(__file__).parents[4] / "models_artifacts"

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
        select(Experiment).order_by(Experiment.started_at.desc()).limit(50)
    )
    return result.scalars().all()


@router.get("/results/algo-agent")
async def get_algo_agent_results(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
):
    """Latest AlgoAgent UCB1 experiment results from disk."""
    results_file = EXPERIMENTS_RESULTS_DIR / "algo_agent_results.json"
    if not results_file.exists():
        return []
    try:
        data = json.loads(results_file.read_text())
        data = sorted(data, key=lambda r: r.get("timestamp", ""), reverse=True)
        return data[:limit]
    except Exception:
        return []


@router.get("/results/summary")
async def get_results_summary(current_user: User = Depends(get_current_user)):
    """Aggregate summary of all experiment results on disk."""
    summary: dict = {"files": [], "total_records": 0, "strategies": {}}
    if not EXPERIMENTS_RESULTS_DIR.exists():
        return summary
    for f in EXPERIMENTS_RESULTS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if not isinstance(data, list):
                data = [data]
            summary["files"].append({"name": f.name, "records": len(data)})
            summary["total_records"] += len(data)
            for rec in data:
                strat = rec.get("strategy", "unknown")
                sharpe = rec.get("sharpe")
                if strat not in summary["strategies"]:
                    summary["strategies"][strat] = {"runs": 0, "best_sharpe": None, "avg_sharpe": 0.0, "sharpes": []}
                summary["strategies"][strat]["runs"] += 1
                if isinstance(sharpe, (int, float)):
                    summary["strategies"][strat]["sharpes"].append(float(sharpe))
        except Exception:
            continue
    for strat, info in summary["strategies"].items():
        sharpes = [s for s in info.pop("sharpes") if s > 0]
        if sharpes:
            info["best_sharpe"] = max(sharpes)
            info["avg_sharpe"] = sum(sharpes) / len(sharpes)
    return summary


@router.get("/checkpoints")
async def list_model_checkpoints(current_user: User = Depends(get_current_user)):
    """List saved model checkpoint files."""
    checkpoints = []
    if MODELS_ARTIFACTS_DIR.exists():
        for f in sorted(MODELS_ARTIFACTS_DIR.rglob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
            checkpoints.append({
                "name": f.name,
                "path": str(f.relative_to(MODELS_ARTIFACTS_DIR)),
                "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                "modified_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
        for f in sorted(MODELS_ARTIFACTS_DIR.rglob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True):
            checkpoints.append({
                "name": f.name,
                "path": str(f.relative_to(MODELS_ARTIFACTS_DIR)),
                "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                "modified_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
    return {"checkpoints": checkpoints, "count": len(checkpoints)}


@router.get("/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
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
