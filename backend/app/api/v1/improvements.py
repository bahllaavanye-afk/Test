"""Self-improvement history endpoint."""
import logging
import time
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/improvements", tags=["improvements"])


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    start_time = time.time()
    from app.main import app
    improver = getattr(app.state, "self_improver", None)
    if improver:
        history = improver.get_history()
    else:
        history = []
    duration = time.time() - start_time
    signal_count = len(history) if isinstance(history, (list, dict)) else 0
    pnl = getattr(improver, "pnl", None) if improver else None
    logger.info(
        f"GET /improvements/history | user_id={getattr(current_user, 'id', 'unknown')} "
        f"| signals={signal_count} | exec_time_ms={duration * 1000:.2f} | pnl={pnl}"
    )
    return history


@router.get("/quality")
async def get_quality(current_user: User = Depends(get_current_user)):
    start_time = time.time()
    from app.main import app
    loop_ref = getattr(app.state, "code_quality_loop", None)
    if loop_ref is None:
        result = {"status": "not_running", "message": "Code quality loop not started"}
    else:
        result = loop_ref.latest()
    duration = time.time() - start_time
    signal_count = (
        len(result.get("metrics", []))
        if isinstance(result, dict) and "metrics" in result
        else 0
    )
    pnl = getattr(loop_ref, "pnl", None) if loop_ref else None
    logger.info(
        f"GET /improvements/quality | user_id={getattr(current_user, 'id', 'unknown')} "
        f"| signals={signal_count} | exec_time_ms={duration * 1000:.2f} | pnl={pnl}"
    )
    return result


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)):
    start_time = time.time()
    from app.main import app
    improver = getattr(app.state, "self_improver", None)
    if improver is None:
        result = {"status": "not_running", "best_params": {}}
    else:
        result = {"best_params": getattr(improver, "_best_params", {})}
    duration = time.time() - start_time
    signal_count = (
        len(result.get("best_params", {}))
        if isinstance(result, dict) and "best_params" in result
        else 0
    )
    pnl = getattr(improver, "pnl", None) if improver else None
    logger.info(
        f"GET /improvements/best_params | user_id={getattr(current_user, 'id', 'unknown')} "
        f"| params_count={signal_count} | exec_time_ms={duration * 1000:.2f} | pnl={pnl}"
    )
    return result