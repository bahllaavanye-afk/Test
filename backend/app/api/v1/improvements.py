"""Self‑improvement history endpoint.

Provides read‑only API routes for retrieving the history of the self‑improver,
the latest code‑quality loop status, and the best parameters discovered by
the improvement process. All endpoints require an authenticated user.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/improvements", tags=["improvements"])


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)) -> List[Any]:
    """Return the full self‑improvement history.

    Retrieves the ``self_improver`` instance from the FastAPI application state
    and returns its stored history. If the improver is not available, an empty
    list is returned.

    Args:
        current_user: The authenticated user, injected via dependency.

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
    """Return the latest code‑quality loop status.

    Accesses the ``code_quality_loop`` stored in the application state and
    returns its most recent status dictionary. If the loop has not been started,
    a status indicating that it is not running is returned.

    Args:
        current_user: The authenticated user, injected via dependency.

    Returns:
        A dictionary with the loop status and optional message.
    """
    from app.main import app

    loop_ref = getattr(app.state, "code_quality_loop", None)
    if loop_ref is None:
        return {"status": "not_running", "message": "Code quality loop not started"}
    return loop_ref.latest()


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Return the best parameters discovered by the self‑improver.

    Retrieves the ``_best_params`` attribute from the ``self_improver`` instance,
    if it exists. If the improver is not running, a status indicating that it is
    not running is returned alongside an empty parameters dictionary.

    Args:
        current_user: The authenticated user, injected via dependency.

    Returns:
        A dictionary containing the best parameters or an empty dict if not
        available.
    """
    from app.main import app

    improver = getattr(app.state, "self_improver", None)
    if improver is None:
        return {"status": "not_running", "best_params": {}}
    return {"best_params": getattr(improver, "_best_params", {})}