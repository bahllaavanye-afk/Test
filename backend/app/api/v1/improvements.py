"""Self-improvement history endpoint."""
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/improvements", tags=["improvements"])

logger = logging.getLogger(__name__)


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
        logger.error(
            "Signal validation failed: expected list, got %s",
            type(raw_signals).__name__,
        )
        raise HTTPException(status_code=400, detail="Invalid signal format")
    if not all(isinstance(sig, dict) for sig in raw_signals):
        logger.error("Signal validation failed: list contains non-dict elements")
        raise HTTPException(status_code=400, detail="Signal items must be dictionaries")
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
    try:
        raw_signals = getattr(improver, "latest_signals", [])
    except Exception as exc:
        logger.exception("Failed to access latest_signals on improver: %s", exc)
        raise HTTPException(status_code=500, detail="Improver signal retrieval error")
    validated = _validate_signals(raw_signals)
    return _process_signals(validated)


def _fetch_filtered_signals(improver: Any) -> List[Dict[str, Any]]:
    """Fetch and filter signals using the existing retrieval helper."""
    return _retrieve_and_filter_signals(improver)


def _format_signal_quality_response(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Construct the response payload for signal quality endpoint."""
    return {"filtered_signals": signals, "count": len(signals)}


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    improver = _get_improver()
    if improver:
        try:
            return improver.get_history()
        except Exception as exc:
            logger.exception(
                "Error retrieving history for user %s: %s",
                getattr(current_user, "id", "unknown"),
                exc,
            )
            raise HTTPException(status_code=500, detail="Failed to retrieve history")
    return []


@router.get("/quality")
async def get_quality(current_user: User = Depends(get_current_user)):
    from app.main import app

    loop_ref = getattr(app.state, "code_quality_loop", None)
    if loop_ref is None:
        return {"status": "not_running", "message": "Code quality loop not started"}
    try:
        return loop_ref.latest()
    except Exception as exc:
        logger.exception(
            "Error fetching code quality loop for user %s: %s",
            getattr(current_user, "id", "unknown"),
            exc,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch code quality")


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)):
    improver = _get_improver()
    if improver is None:
        return {"status": "not_running", "best_params": {}}
    try:
        return {"best_params": getattr(improver, "_best_params", {})}
    except Exception as exc:
        logger.exception(
            "Error retrieving best_params for user %s: %s",
            getattr(current_user, "id", "unknown"),
            exc,
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve best parameters")


@router.get("/signal_quality")
async def get_signal_quality(current_user: User = Depends(get_current_user)):
    """
    Return signals after applying tightened entry conditions and improved exit logic.
    """
    improver = _get_improver()
    if improver is None:
        raise HTTPException(status_code=404, detail="Improver not initialized")
    try:
        filtered_signals = _fetch_filtered_signals(improver)
        return _format_signal_quality_response(filtered_signals)
    except HTTPException:
        # Propagate HTTPExceptions raised in helper functions unchanged
        raise
    except Exception as exc:
        logger.exception(
            "Unexpected error processing signal quality for user %s: %s",
            getattr(current_user, "id", "unknown"),
            exc,
        )
        raise HTTPException(status_code=500, detail="Failed to process signal quality")