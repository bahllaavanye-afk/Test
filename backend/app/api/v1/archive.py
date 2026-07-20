"""Trade archive replay endpoints."""
from fastapi import APIRouter, Depends, Query, HTTPException
from app.api.deps import get_current_user
from app.models.user import User
from app.archive.trade_archiver import replay, list_archives

router = APIRouter(prefix="/archive", tags=["archive"])


@router.get("/index")
async def get_index(current_user: User = Depends(get_current_user)):
    """
    Return a list of available archives.
    Handles the case where the underlying function returns None.
    """
    archives = list_archives()
    # Ensure a list is always returned
    return archives if archives else []


@router.get("/{category}")
async def get_archive(
    category: str,
    date: str | None = Query(
        None, description="YYYY-MM-DD, defaults to today"
    ),
    limit: int = Query(
        500,
        ge=1,
        le=5000,
        description="Maximum number of records to return (1-5000)",
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Replay trades for a given category and optional date.
    Edge‑case handling:
    * `date` empty string is treated as None.
    * `limit` is validated to be at least 1.
    * Returns an empty list if the replay yields no data.
    """
    # Normalize empty date strings
    if date == "":
        date = None

    # Defensive check for limit (should already be enforced by Query)
    if limit < 1:
        raise HTTPException(
            status_code=400,
            detail="Limit must be a positive integer.",
        )

    try:
        result = replay(category, date, limit)
    except Exception as exc:
        # Convert unexpected errors to a client‑friendly response
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve archive: {exc}",
        ) from exc

    # Ensure the endpoint always returns a list
    return result if result else []