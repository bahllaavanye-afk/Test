import logging
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/improvements", tags=["improvements"])


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    from app.main import app
    improver = getattr(app.state, "self_improver", None)
    if improver:
        history = improver.get_history()
        signal_count = len(history) if history else 0
        logger.info(
            "Retrieved improvement history",
            extra={
                "signal_count": signal_count,
                "user_id": current_user.id,
            }
        )
        return history
    logger.info(
        "Self-improver not available",
        extra={"user_id": current_user.id}
    )
    return []


@router.get("/quality")
async def get_quality(current_user: User = Depends(get_current_user)):
    from app.main import app
    loop_ref = getattr(app.state, "code_quality_loop", None)
    if loop_ref is None:
        logger.info(
            "Code quality loop not running",
            extra={"user_id": current_user.id}
        )
        return {"status": "not_running", "message": "Code quality loop not started"}
    result = loop_ref.latest()
    logger.info(
        "Retrieved code quality metrics",
        extra={
            "status": result.get("status"),
            "user_id": current_user.id,
        }
    )
    return result


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)):
    from app.main import app
    improver = getattr(app.state, "self_improver", None)
    if improver is None:
        logger.info(
            "Self-improver not available for best params",
            extra={"user_id": current_user.id}
        )
        return {"status": "not_running", "best_params": {}}
    best_params = getattr(improver, "_best_params", {})
    param_count = len(best_params) if best_params else 0
    logger.info(
        "Retrieved best parameters",
        extra={
            "param_count": param_count,
            "user_id": current_user.id,
        }
    )
    return {"best_params": best_params}