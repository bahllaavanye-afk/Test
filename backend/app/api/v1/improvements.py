"""Self-improvement history endpoint."""
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/improvements", tags=["improvements"])


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
    if loop_ref:
        return loop_ref.latest()
    return None


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)):
    from app.main import app
    improver = getattr(app.state, "self_improver", None)
    if improver:
        return improver._best_params
    return {}
