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
    from app.main import app
    agent = getattr(app.state, "algo_agent", None)
    if not agent:
        return {"running": False}
    return {
        "running": agent._running,
        "total_runs": agent._total_runs,
        "candidates": len(agent._candidates),
        "top_3": agent.get_leaderboard()[:3],
    }
