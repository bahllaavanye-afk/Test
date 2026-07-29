"""Self-improvement history endpoint."""
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User

# Endpoint path constants
HISTORY_PATH = "/history"
QUALITY_PATH = "/quality"
BEST_PARAMS_PATH = "/best_params"

# Response status/message constants
STATUS_NOT_RUNNING = "not_running"
MSG_QUALITY_NOT_RUNNING = "Code quality loop not started"

router = APIRouter(prefix="/improvements", tags=["improvements"])


@router.get(HISTORY_PATH)
async def get_history(current_user: User = Depends(get_current_user)):
    from app.main import app
    improver = getattr(app.state, "self_improver", None)
    if improver:
        return improver.get_history()
    return []


@router.get(QUALITY_PATH)
async def get_quality(current_user: User = Depends(get_current_user)):
    from app.main import app
    loop_ref = getattr(app.state, "code_quality_loop", None)
    if loop_ref is None:
        return {"status": STATUS_NOT_RUNNING, "message": MSG_QUALITY_NOT_RUNNING}
    return loop_ref.latest()


@router.get(BEST_PARAMS_PATH)
async def get_best_params(current_user: User = Depends(get_current_user)):
    from app.main import app
    improver = getattr(app.state, "self_improver", None)
    if improver is None:
        return {"status": STATUS_NOT_RUNNING, "best_params": {}}
    return {"best_params": getattr(improver, "_best_params", {})}