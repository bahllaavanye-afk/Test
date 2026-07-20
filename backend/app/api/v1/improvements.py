"""Self-improvement history endpoint."""
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/improvements", tags=["improvements"])


def _get_self_improver():
    """Retrieve the self-improver instance from app state."""
    from app.main import app
    return getattr(app.state, "self_improver", None)


def _get_code_quality_loop():
    """Retrieve the code quality loop instance from app state."""
    from app.main import app
    return getattr(app.state, "code_quality_loop", None)


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    improver = _get_self_improver()
    if improver:
        return improver.get_history()
    return []


@router.get("/quality")
async def get_quality(current_user: User = Depends(get_current_user)):
    loop_ref = _get_code_quality_loop()
    if loop_ref is None:
        return {"status": "not_running", "message": "Code quality loop not started"}
    return loop_ref.latest()


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)):
    improver = _get_self_improver()
    if improver is None:
        return {"status": "not_running", "best_params": {}}
    return {"best_params": getattr(improver, "_best_params", {})}