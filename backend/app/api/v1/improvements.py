"""Self-improvement history endpoint."""
from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_current_user
from app.models.user import User
from typing import List, Dict, Any

router = APIRouter(prefix="/improvements", tags=["improvements"])


def _filter_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply tightened entry conditions and improved exit logic in a single pass.
    Expected signal fields:
        - entry_score: float (0-1)
        - volume: float
        - avg_volume: float
        - ma_cross: bool (moving‑average crossover confirmation)
        - profit_target_hit: bool
        - trailing_stop_triggered: bool
    """
    return [
        sig
        for sig in signals
        if sig.get("entry_score", 0) >= 0.7
        and sig.get("volume", 0) >= sig.get("avg_volume", 0) * 1.2
        and sig.get("ma_cross", False)
        and (sig.get("profit_target_hit") or sig.get("trailing_stop_triggered"))
    ]


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    from app.main import app
    improver = getattr(app.state, "self_improver", None)
    if improver:
        return improver.get_history()
    return []


@router.get("/quality")
async def get_quality(current_user: User = Depends(get_current_user)):
    from app.main import app
    loop_ref = getattr(app.state, "code_quality_loop", None)
    if loop_ref is None:
        return {"status": "not_running", "message": "Code quality loop not started"}
    return loop_ref.latest()


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)):
    from app.main import app
    improver = getattr(app.state, "self_improver", None)
    if improver is None:
        return {"status": "not_running", "best_params": {}}
    return {"best_params": getattr(improver, "_best_params", {})}


@router.get("/signal_quality")
async def get_signal_quality(current_user: User = Depends(get_current_user)):
    """
    Return signals after applying tightened entry conditions and improved exit logic.
    """
    from app.main import app
    improver = getattr(app.state, "self_improver", None)
    if improver is None:
        raise HTTPException(status_code=404, detail="Improver not initialized")
    raw_signals = getattr(improver, "latest_signals", [])
    if not isinstance(raw_signals, list):
        raise HTTPException(status_code=500, detail="Invalid signal format")
    if not raw_signals:
        return {"filtered_signals": [], "count": 0}
    final_signals = _filter_signals(raw_signals)
    return {"filtered_signals": final_signals, "count": len(final_signals)}