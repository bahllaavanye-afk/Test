"""Self-improvement history endpoint."""
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/improvements", tags=["improvements"])

logger = logging.getLogger(__name__)


def _ensure_list(value: Any) -> List[Any]:
    """Return a list representation, handling None or non‑list inputs."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    # For unexpected types, wrap in a list to avoid breaking callers.
    return [value]


def _ensure_dict(value: Any) -> Dict[Any, Any]:
    """Return a dict representation, handling None or non‑dict inputs."""
    if isinstance(value, dict):
        return value
    return {} if value is None else {"value": value}


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    """Return the improvement history safely handling edge cases."""
    from app.main import app

    improver = getattr(app.state, "self_improver", None)
    if improver is None:
        return []

    try:
        raw_history = improver.get_history()
        return _ensure_list(raw_history)
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to retrieve improvement history: %s", exc)
        return []


@router.get("/quality")
async def get_quality(current_user: User = Depends(get_current_user)):
    """Return the latest code quality information, guarding against missing data."""
    from app.main import app

    loop_ref = getattr(app.state, "code_quality_loop", None)
    if loop_ref is None:
        return {"status": "not_running", "message": "Code quality loop not started"}

    try:
        latest = loop_ref.latest()
        # Ensure the result is a dict; if it's a list, return the last element safely.
        if isinstance(latest, dict):
            return latest
        if isinstance(latest, list) and latest:
            return _ensure_dict(latest[-1])
        # Fallback for unexpected types or empty list.
        return {"status": "unknown", "message": "No quality data available"}
    except Exception as exc:  # pragma: no cover
        logger.exception("Error retrieving code quality info: %s", exc)
        return {"status": "error", "message": str(exc)}


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)):
    """Return the best parameters discovered, handling None/empty cases."""
    from app.main import app

    improver = getattr(app.state, "self_improver", None)
    if improver is None:
        return {"status": "not_running", "best_params": {}}

    try:
        best_params = getattr(improver, "_best_params", {})
        return {"best_params": _ensure_dict(best_params)}
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to retrieve best params: %s", exc)
        return {"status": "error", "best_params": {}}