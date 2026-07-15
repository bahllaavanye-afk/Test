"""Trade archive replay endpoints."""
from datetime import datetime

from fastapi import APIRouter, Depends, Query, HTTPException, status

from app.api.deps import get_current_user
from app.models.user import User
from app.archive.trade_archiver import replay, list_archives

router = APIRouter(prefix="/archive", tags=["archive"])


@router.get("/index")
async def get_index(current_user: User = Depends(get_current_user)):
    """Return a list of available archive categories."""
    return list_archives()


@router.get("/{category}")
async def get_archive(
    category: str,
    date: str | None = Query(
        None,
        description="YYYY-MM-DD, defaults to today",
        regex=r"^\d{4}-\d{2}-\d{2}$",
    ),
    limit: int = Query(500, le=5000, description="Maximum number of records to return"),
    current_user: User = Depends(get_current_user),
):
    """
    Replay trades from a specific archive category.

    Parameters
    ----------
    category: str
        Archive category to replay.
    date: str | None
        Optional date in YYYY-MM-DD format. If omitted, defaults to today.
    limit: int
        Maximum number of records to return (default 500, max 5000).
    current_user: User
        Authenticated user (injected by dependency).

    Returns
    -------
    The replayed trade data for the requested category and date.
    """
    if date is not None:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format, expected YYYY-MM-DD",
            ) from exc

    return replay(category, date, limit)