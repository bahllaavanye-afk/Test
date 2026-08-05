"""Self-improvement history endpoint.

Provides read‑only access to the internal self‑improvement components
used by the platform, such as the improvement history, the code‑quality
loop status, and the best hyper‑parameters discovered so far.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/improvements", tags=["improvements"])


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)) -> List[Any]:
    """Return the historical record of self‑improvement actions.

    The endpoint retrieves the ``self_improver`` instance attached to the
    FastAPI application state and returns its history. If the improver
    is not present, an empty list is returned.

    Args:
        current_user: The authenticated user obtained via dependency injection.

    Returns:
        A list containing the improvement history entries.
    """
    from app.main import app

    improver = getattr(app.state, "self_improver", None)
    if improver:
        return improver.get_history()
    return []


@router.get("/quality")
async def get_quality(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Provide the current status of the code‑quality monitoring loop.

    If the ``code_quality_loop`` is not running, a status payload indicating
    that condition is returned.

    Args:
        current_user: The authenticated user obtained via dependency injection.

    Returns:
        A dictionary with the loop status and any additional information.
    """
    from app.main import app

    loop_ref = getattr(app.state, "code_quality_loop", None)
    if loop_ref is None:
        return {"status": "not_running", "message": "Code quality loop not started"}
    return loop_ref.latest()


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Retrieve the best hyper‑parameters discovered by the self‑improver.

    If the ``self_improver`` instance is unavailable, a payload indicating
    that the process is not running is returned.

    Args:
        current_user: The authenticated user obtained via dependency injection.

    Returns:
        A dictionary containing the best parameters or an empty mapping if
        the improver is not active.
    """
    from app.main import app

    improver = getattr(app.state, "self_improver", None)
    if improver is None:
        return {"status": "not_running", "best_params": {}}
    return {"best_params": getattr(improver, "_best_params", {})}