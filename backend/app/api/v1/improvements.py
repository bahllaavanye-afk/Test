"""Self-improvement history endpoint."""
import logging
from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_current_user
from app.models.user import User
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

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
    filtered = []
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
    refined = []
    for sig in signals:
        if sig.get("profit_target_hit") or sig.get("trailing_stop_triggered"):
            refined.append(sig)
    return refined


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    try:
        from app.main import app
        improver = getattr(app.state, "self_improver", None)
        if improver:
            return improver.get_history()
        return []
    except AttributeError as exc:
        logger.exception("Failed to access application state for history endpoint")
        raise HTTPException(status_code=500, detail="Application state error") from exc
    except Exception as exc:
        logger.exception("Unexpected error in get_history")
        raise HTTPException(status_code=500, detail="Unexpected server error") from exc


@router.get("/quality")
async def get_quality(current_user: User = Depends(get_current_user)):
    try:
        from app.main import app
        loop_ref = getattr(app.state, "code_quality_loop", None)
        if loop_ref is None:
            return {"status": "not_running", "message": "Code quality loop not started"}
        return loop_ref.latest()
    except AttributeError as exc:
        logger.exception("Failed to retrieve code_quality_loop")
        raise HTTPException(status_code=500, detail="Application state error") from exc
    except Exception as exc:
        logger.exception("Unexpected error in get_quality")
        raise HTTPException(status_code=500, detail="Unexpected server error") from exc


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)):
    try:
        from app.main import app
        improver = getattr(app.state, "self_improver", None)
        if improver is None:
            return {"status": "not_running", "best_params": {}}
        return {"best_params": getattr(improver, "_best_params", {})}
    except AttributeError as exc:
        logger.exception("Failed to access self_improver for best_params")
        raise HTTPException(status_code=500, detail="Application state error") from exc
    except Exception as exc:
        logger.exception("Unexpected error in get_best_params")
        raise HTTPException(status_code=500, detail="Unexpected server error") from exc


@router.get("/signal_quality")
async def get_signal_quality(current_user: User = Depends(get_current_user)):
    """
    Return signals after applying tightened entry conditions and improved exit logic.
    """
    try:
        from app.main import app
        improver = getattr(app.state, "self_improver", None)
        if improver is None:
            raise HTTPException(status_code=404, detail="Improver not initialized")
        raw_signals = getattr(improver, "latest_signals", [])
        if not isinstance(raw_signals, list):
            raise HTTPException(status_code=500, detail="Invalid signal format")
        # Apply entry filters then exit logic
        entry_filtered = _apply_entry_filters(raw_signals)
        final_signals = _apply_exit_logic(entry_filtered)
        return {"filtered_signals": final_signals, "count": len(final_signals)}
    except HTTPException:
        # Already logged or client-facing; re-raise unchanged
        raise
    except AttributeError as exc:
        logger.exception("Attribute error while processing signal_quality")
        raise HTTPException(status_code=500, detail="Application state error") from exc
    except Exception as exc:
        logger.exception("Unexpected error in get_signal_quality")
        raise HTTPException(status_code=500, detail="Unexpected server error") from exc