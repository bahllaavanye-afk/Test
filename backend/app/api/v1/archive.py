"""Trade archive replay endpoints."""
from fastapi import APIRouter, Depends, Query

from app.api.deps import get_current_user
from app.archive.trade_archiver import list_archives, replay
from app.models.user import User

router = APIRouter(prefix="/archive", tags=["archive"])


@router.get("/index")
async def get_index(current_user: User = Depends(get_current_user)):
    return list_archives()


@router.get("/{category}")
async def get_archive(
    category: str,
    date: str | None = Query(None, description="YYYY-MM-DD, defaults to today"),
    limit: int = Query(500, le=5000),
    current_user: User = Depends(get_current_user),
):
    return replay(category, date, limit)
