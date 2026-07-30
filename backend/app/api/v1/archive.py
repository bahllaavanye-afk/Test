"""Trade archive replay endpoints."""
from fastapi import APIRouter, Depends, Query, HTTPException
from app.api.deps import get_current_user
from app.models.user import User
from app.archive.trade_archiver import replay, list_archives

# Constants
ARCHIVE_PREFIX: str = "/archive"
ARCHIVE_TAG: str = "archive"

DEFAULT_LIMIT: int = 500
MIN_LIMIT: int = 1
MAX_LIMIT: int = 5000

DATE_DESCRIPTION: str = "YYYY-MM-DD, defaults to today"
LIMIT_DESCRIPTION: str = "Maximum number of records to return (1-5000)"

ERR_LIMIT_POSITIVE: str = "Limit must be a positive integer."
ERR_RETRIEVE_ARCHIVE: str = "Failed to retrieve archive: {exc}"
ERR_CATEGORY_REQUIRED: str = "Category must be a non‑empty string."

router = APIRouter(prefix=ARCHIVE_PREFIX, tags=[ARCHIVE_TAG])


@router.get("/index")
async def get_index(current_user: User = Depends(get_current_user)):
    """
    Return a list of available archives.
    Handles the case where the underlying function returns None or a non‑iterable.
    """
    archives = list_archives()
    if not isinstance(archives, (list, tuple, set)):
        return []
    return list(archives)


@router.get("/{category}")
async def get_archive(
    category: str,
    date: str | None = Query(
        None, description=DATE_DESCRIPTION
    ),
    limit: int | None = Query(
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
    * `category` must be a non‑empty string.
    * Empty `date` strings are treated as None.
    * `limit` defaults to DEFAULT_LIMIT if None and is validated to be within bounds.
    * Returns an empty list if the replay yields no data or a non‑iterable result.
    """
    # Validate category
    if not category or not isinstance(category, str):
        raise HTTPException(status_code=400, detail=ERR_CATEGORY_REQUIRED)

    # Normalize empty date strings
    if isinstance(date, str) and date.strip() == "":
        date = None

    # Defensive handling for limit
    if limit is None:
        limit = DEFAULT_LIMIT
    if limit < MIN_LIMIT:
        raise HTTPException(status_code=400, detail=ERR_LIMIT_POSITIVE)

    try:
        result = replay(category, date, limit)
    except Exception as exc:
        # Convert unexpected errors to a client‑friendly response
        raise HTTPException(
            status_code=500,
            detail=ERR_RETRIEVE_ARCHIVE.format(exc=exc),
        ) from exc

    # Ensure the endpoint always returns a list
    if not isinstance(result, (list, tuple, set)):
        return []
    return list(result)