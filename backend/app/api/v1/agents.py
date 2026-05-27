"""AlgoAgent monitoring endpoints."""
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User
import json
from pathlib import Path

router = APIRouter(prefix="/agents", tags=["agents"])

RESULTS_FILE = Path(__file__).parents[4] / "experiments" / "results" / "algo_agent_results.json"


@router.get("/leaderboard")
async def get_leaderboard(current_user: User = Depends(get_current_user)):
    """Current UCB1 leaderboard — best strategies by avg Sharpe."""
    from app.main import app
    agent = getattr(app.state, "algo_agent", None)
    if agent:
        return agent.get_leaderboard()
    return []


@router.get("/results")
async def get_results(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
):
    """Recent AlgoAgent backtest results."""
    if not RESULTS_FILE.exists():
        return []
    try:
        data = json.loads(RESULTS_FILE.read_text())
        return sorted(data, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
    except Exception:
        return []


@router.get("/status")
async def agent_status(current_user: User = Depends(get_current_user)):
    """Status of ALL background agents."""
    from app.main import app

    algo_agent = getattr(app.state, "algo_agent", None)
    self_improver = getattr(app.state, "self_improver", None)
    qa_monitor = getattr(app.state, "qa_monitor", None)
    research_scientist = getattr(app.state, "research_scientist", None)
    modeling_engineer = getattr(app.state, "modeling_engineer", None)

    return {
        "algo_agent": {
            "running": getattr(algo_agent, "_running", False),
            "total_runs": getattr(algo_agent, "_total_runs", 0),
            "candidates": len(getattr(algo_agent, "_candidates", {})),
            "top_3": algo_agent.get_leaderboard()[:3] if algo_agent else [],
        },
        "self_improver": {
            "running": getattr(self_improver, "_running", False),
            "iteration": getattr(self_improver, "_iteration", 0),
        },
        "qa_monitor": {
            "running": getattr(qa_monitor, "_running", False),
        },
        "research_scientist": {
            "running": research_scientist is not None,
            "cycles_completed": getattr(research_scientist, "_cycle", 0),
            "total_findings": len(getattr(research_scientist, "_findings", [])),
        },
        "modeling_engineer": {
            "running": modeling_engineer is not None,
            "cycles_completed": getattr(modeling_engineer, "_cycle", 0),
            "decisions_made": len(getattr(modeling_engineer, "_decisions", [])),
        },
    }


@router.get("/research")
async def get_research_summary(current_user: User = Depends(get_current_user)):
    """Research Scientist findings — top alpha ideas and experiment queue."""
    from app.main import app
    agent = getattr(app.state, "research_scientist", None)
    if not agent:
        return {"error": "ResearchScientist not running", "cycles_completed": 0, "top_ideas": []}
    return agent.get_research_summary()


@router.get("/modeling")
async def get_modeling_summary(current_user: User = Depends(get_current_user)):
    """Modeling Engineer summary — model health, drift, and recent decisions."""
    from app.main import app
    agent = getattr(app.state, "modeling_engineer", None)
    if not agent:
        return {"error": "ModelingEngineer not running", "cycles_completed": 0, "models_monitored": []}
    return agent.get_engineering_summary()
