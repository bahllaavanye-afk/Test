"""Self-improvement history endpoint."""
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/improvements", tags=["improvements"])


def _safe_app_state(attr: str, default=None):
    """Safely read an attribute from app.state, returning default if not set."""
    try:
        from app.main import app
        state = getattr(app, "state", None)
        if state is None:
            return default
        return getattr(state, attr, default)
    except Exception:
        return default


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    improver = _safe_app_state("self_improver")
    if improver is not None:
        try:
            return improver.get_history()
        except Exception:
            pass
    return []


@router.get("/quality")
async def get_quality(current_user: User = Depends(get_current_user)):
    loop_ref = _safe_app_state("code_quality_loop")
    if loop_ref is not None:
        try:
            result = loop_ref.latest()
            if result is not None:
                return result
        except Exception:
            pass
    return {"status": "unavailable", "metrics": {}}


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)):
    improver = _safe_app_state("self_improver")
    if improver is not None:
        try:
            return improver._best_params or {}
        except Exception:
            pass
    return {}
