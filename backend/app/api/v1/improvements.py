"""Self-improvement history endpoint."""
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/improvements", tags=["improvements"])


def _apply_entry_filters(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tighten entry conditions and add confirmation filters.

    Expected signal fields:
        - entry_score: float (0-1)
        - volume: float
        - avg_volume: float
        - ma_cross: bool (moving‑average crossover confirmation)
    """
    filtered: List[Dict[str, Any]] = []
    for sig in signals:
        # Basic score threshold
        if sig.get("entry_score", 0) < 0.7:
            continue
        # Volume confirmation (at least 20% above average)
        if sig.get("volume", 0) < sig.get("avg_volume", 0) * 1.2:
            continue
        # Moving‑average crossover confirmation
        if not sig.get("ma_cross", False):
            continue
        filtered.append(sig)
    return filtered


def _apply_exit_logic(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Improve exit logic by enforcing either a profit target or a trailing stop.

    Expected signal fields:
        - profit_target_hit: bool
        - trailing_stop_triggered: bool

    Signals that satisfy either condition are kept; otherwise they are removed.
    """
    refined: List[Dict[str, Any]] = []
    for sig in signals:
        if sig.get("profit_target_hit") or sig.get("trailing_stop_triggered"):
            refined.append(sig)
    return refined


def _get_improver() -> Any:
    """Retrieve the self_improver instance from the global app state."""
    from app.main import app

    return getattr(app.state, "self_improver", None)


def _validate_signals(raw_signals: Any) -> List[Dict[str, Any]]:
    """
    Ensure raw_signals is a list of dictionaries.
    Raises HTTPException on validation failure.
    """
    if not isinstance(raw_signals, list):
        raise HTTPException(status_code=500, detail="Invalid signal format")
    return raw_signals


def _process_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply entry filters followed by exit logic."""
    entry_filtered = _apply_entry_filters(signals)
    return _apply_exit_logic(entry_filtered)


def _retrieve_and_filter_signals(improver: Any) -> List[Dict[str, Any]]:
    """
    Helper to fetch the latest signals from the improver,
    validate them, and apply entry/exit filters.
    """
    raw_signals = getattr(improver, "latest_signals", [])
    validated = _validate_signals(raw_signals)
    return _process_signals(validated)


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    improver = _get_improver()
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
    improver = _get_improver()
    if improver is None:
        return {"status": "not_running", "best_params": {}}
    return {"best_params": getattr(improver, "_best_params", {})}


@router.get("/signal_quality")
async def get_signal_quality(current_user: User = Depends(get_current_user)):
    """
    Return signals after applying tightened entry conditions and improved exit logic.
    """
    improver = _get_improver()
    if improver is None:
        raise HTTPException(status_code=404, detail="Improver not initialized")
    final_signals = _retrieve_and_filter_signals(improver)
    return {"filtered_signals": final_signals, "count": len(final_signals)}