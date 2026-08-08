"""Self-improvement history endpoint."""
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user
from app.models.user import User

# Constants
ENTRY_SCORE_THRESHOLD = 0.7
VOLUME_MULTIPLIER = 1.2

STATUS_NOT_RUNNING = "not_running"
DETAIL_IMPROVER_NOT_INITIALIZED = "Improver not initialized"
DETAIL_HISTORY_RETRIEVAL_FAILED = "Failed to retrieve history"
DETAIL_CODE_QUALITY_NOT_STARTED = "Code quality loop not started"
DETAIL_CODE_QUALITY_FETCH_FAILED = "Failed to fetch code quality"
DETAIL_BEST_PARAMS_NOT_RUNNING = "not_running"
DETAIL_BEST_PARAMS_FETCH_FAILED = "Failed to retrieve best parameters"
DETAIL_PROCESS_SIGNAL_QUALITY_FAILED = "Failed to process signal quality"
ERROR_MSG_INVALID_FORMAT = "Invalid signal format"
ERROR_MSG_NON_DICT = "Signal items must be dictionaries"
LOG_MSG_VALIDATION_TYPE = "Signal validation failed: expected list, got %s"
LOG_MSG_VALIDATION_NON_DICT = "Signal validation failed: list contains non-dict elements"

# Additional string constants for logging and error details
DETAIL_IMPROVER_RETRIEVAL_ERROR = "Improver signal retrieval error"
LOG_MSG_IMPROVER_ACCESS_FAILED = "Failed to access latest_signals on improver: %s"
LOG_MSG_HISTORY_ERROR = "Error retrieving history for user %s: %s"
LOG_MSG_QUALITY_ERROR = "Error fetching code quality loop for user %s: %s"
LOG_MSG_BEST_PARAMS_ERROR = "Error retrieving best_params for user %s: %s"
LOG_MSG_SIGNAL_QUALITY_UNEXPECTED = "Unexpected error processing signal quality for user %s: %s"

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
        if sig.get("entry_score", 0) < ENTRY_SCORE_THRESHOLD:
            continue
        # Volume confirmation (at least VOLUME_MULTIPLIER above average)
        if sig.get("volume", 0) < sig.get("avg_volume", 0) * VOLUME_MULTIPLIER:
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
        logger.error(LOG_MSG_VALIDATION_TYPE, type(raw_signals).__name__)
        raise HTTPException(status_code=400, detail=ERROR_MSG_INVALID_FORMAT)
    if not all(isinstance(sig, dict) for sig in raw_signals):
        logger.error(LOG_MSG_VALIDATION_NON_DICT)
        raise HTTPException(status_code=400, detail=ERROR_MSG_NON_DICT)
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
        logger.exception(LOG_MSG_IMPROVER_ACCESS_FAILED, exc)
        raise HTTPException(status_code=500, detail=DETAIL_IMPROVER_RETRIEVAL_ERROR)
    validated = _validate_signals(raw_signals)
    return _process_signals(validated)


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    improver = _get_improver()
    if improver:
        try:
            return improver.get_history()
        except Exception as exc:
            logger.exception(
                LOG_MSG_HISTORY_ERROR,
                getattr(current_user, "id", "unknown"),
                exc,
            )
            raise HTTPException(status_code=500, detail=DETAIL_HISTORY_RETRIEVAL_FAILED)
    return []


@router.get("/quality")
async def get_quality(current_user: User = Depends(get_current_user)):
    from app.main import app

    loop_ref = getattr(app.state, "code_quality_loop", None)
    if loop_ref is None:
        return {"status": STATUS_NOT_RUNNING, "message": DETAIL_CODE_QUALITY_NOT_STARTED}
    try:
        return loop_ref.latest()
    except Exception as exc:
        logger.exception(
            LOG_MSG_QUALITY_ERROR,
            getattr(current_user, "id", "unknown"),
            exc,
        )
        raise HTTPException(status_code=500, detail=DETAIL_CODE_QUALITY_FETCH_FAILED)


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)):
    improver = _get_improver()
    if improver is None:
        return {"status": STATUS_NOT_RUNNING, "best_params": {}}
    try:
        return {"best_params": getattr(improver, "_best_params", {})}
    except Exception as exc:
        logger.exception(
            LOG_MSG_BEST_PARAMS_ERROR,
            getattr(current_user, "id", "unknown"),
            exc,
        )
        raise HTTPException(status_code=500, detail=DETAIL_BEST_PARAMS_FETCH_FAILED)


@router.get("/signal_quality")
async def get_signal_quality(current_user: User = Depends(get_current_user)):
    """
    Return signals after applying tightened entry conditions and improved exit logic.
    """
    improver = _get_improver()
    if improver is None:
        raise HTTPException(status_code=404, detail=DETAIL_IMPROVER_NOT_INITIALIZED)
    try:
        final_signals = _retrieve_and_filter_signals(improver)
        return {"filtered_signals": final_signals, "count": len(final_signals)}
    except HTTPException:
        # Propagate HTTPExceptions raised in helper functions unchanged
        raise
    except Exception as exc:
        logger.exception(
            LOG_MSG_SIGNAL_QUALITY_UNEXPECTED,
            getattr(current_user, "id", "unknown"),
            exc,
        )
        raise HTTPException(status_code=500, detail=DETAIL_PROCESS_SIGNAL_QUALITY_FAILED)