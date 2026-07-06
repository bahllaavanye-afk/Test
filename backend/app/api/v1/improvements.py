from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_current_user
from app.models.user import User
from functools import lru_cache
from typing import Dict

router = APIRouter(prefix="/improvements", tags=["improvements"])


@lru_cache(maxsize=128)
def get_improver(app):
    return getattr(app.state, "self_improver", None)


@lru_cache(maxsize=128)
def get_loop_ref(app):
    return getattr(app.state, "code_quality_loop", None)


@router.get("/history")
async def get_history(current_user: User = Depends(get_current_user)):
    from app.main import app
    improver = get_improver(app)
    if improver is None:
        raise HTTPException(status_code=404, detail="Self improver not found")
    return improver.get_history()


@router.get("/quality")
async def get_quality(current_user: User = Depends(get_current_user)):
    from app.main import app
    loop_ref = get_loop_ref(app)
    if loop_ref is None:
        raise HTTPException(status_code=404, detail="Code quality loop not started")
    return loop_ref.latest()


@router.get("/best_params")
async def get_best_params(current_user: User = Depends(get_current_user)):
    from app.main import app
    improver = get_improver(app)
    if improver is None:
        raise HTTPException(status_code=404, detail="Self improver not found")
    best_params = getattr(improver, "_best_params", {})
    return {"best_params": best_params}