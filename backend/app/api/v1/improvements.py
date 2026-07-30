"""Self-improvement history endpoint."""
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User
from functools import lru_cache

# Import the central FastAPI app once to avoid repeated imports.
# This import is safe assuming app.state is populated after startup.
from app.main import app as main_app

router = APIRouter(prefix="/improvements", tags=["improvements"])


@lru_cache(maxsize=1)
def _cached_history():
    """Return cached self‑improvement history.

    The cache is limited to a single entry because the history is
    expected to change only when the underlying improver updates.
    """
    improver = getattr(main_app.state, "self_improver", None)
    if improver:
        return improver.get_history()
    return []


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    """Endpoint returning the self‑improvement history.

    Uses an in‑memory cache to avoid recomputing the history on every request.
    """
    return _cached_history()


@router.get("/quality")
async def get_quality(current_user: User = Depends(get_current_user)):
    """Endpoint returning the latest code‑quality loop status."""
    loop_ref = getattr(main_app.state, "code_quality_loop", None)
    if loop_ref is None:
        return {"status": "not_running", "message": "Code quality loop not started"}
    return loop_ref.latest()


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)):
    """Endpoint returning the best parameters discovered by the improver."""
    improver = getattr(main_app.state, "self_improver", None)
    if improver is None:
        return {"status": "not_running", "best_params": {}}
    return {"best_params": getattr(improver, "_best_params", {})}