"""Trade archive replay endpoints."""
from fastapi import APIRouter, Depends, Query, HTTPException
from app.api.deps import get_current_user
from app.models.user import User
from app.archive.trade_archiver import replay, list_archives

# Constants
ARCHIVE_PREFIX: str = "/archive"
ARCHIVE_TAG: str = "archive"

INDEX_PATH: str = "/index"
CATEGORY_PATH: str = "/{category}"

DEFAULT_LIMIT: int = 500
MIN_LIMIT: int = 1
MAX_LIMIT: int = 5000

DATE_DESCRIPTION: str = "YYYY-MM-DD, defaults to today"
LIMIT_DESCRIPTION: str = "Maximum number of records to return (1-5000)"

ERR_LIMIT_POSITIVE: str = "Limit must be a positive integer."
ERR_RETRIEVE_ARCHIVE: str = "Failed to retrieve archive: {exc}"

router = APIRouter(prefix=ARCHIVE_PREFIX, tags=[ARCHIVE_TAG])


@router.get(INDEX_PATH)
async def get_index(current_user: User = Depends(get_current_user)):
    """
    Return a list of available archives.
    Handles the case where the underlying function returns None.
    """
    archives = list_archives()
    # Ensure a list is always returned
    return archives if archives else []


@router.get(CATEGORY_PATH)
async def get_archive(
    category: str,
    date: str | None = Query(
        None, description=DATE_DESCRIPTION
    ),
    limit: int = Query(
        DEFAULT_LIMIT,
        ge=MIN_LIMIT,
        le=MAX_LIMIT,
        description=LIMIT_DESCRIPTION,
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
    if limit < MIN_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=ERR_LIMIT_POSITIVE,
        )

    try:
        result = replay(category, date, limit)
    except Exception as exc:
        # Convert unexpected errors to a client‑friendly response
        raise HTTPException(
            status_code=500,
            detail=ERR_RETRIEVE_ARCHIVE.format(exc=exc),
        ) from exc

    # Ensure the endpoint always returns a list
    return result if result else []