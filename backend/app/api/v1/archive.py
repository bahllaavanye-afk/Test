"""Trade archive replay endpoints."""
import logging
from fastapi import APIRouter, Depends, Query, HTTPException
from app.api.deps import get_current_user
from app.models.user import User
from app.archive.trade_archiver import replay, list_archives

# Set up structured logger
logger = logging.getLogger(__name__)

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
ERR_LIST_ARCHIVES: str = "Failed to list archives: {exc}"

router = APIRouter(prefix=ARCHIVE_PREFIX, tags=[ARCHIVE_TAG])


@router.get("/index")
async def get_index(current_user: User = Depends(get_current_user)):
    """
    Return a list of available archives.
    Handles the case where the underlying function returns None.
    """
    try:
        archives = list_archives()
    except Exception as exc:
        logger.error(
            "Error retrieving archive list",
            exc_info=True,
            error=str(exc)
        )
        raise HTTPException(
            status_code=500,
            detail=ERR_LIST_ARCHIVES.format(exc=exc),
        ) from exc

    # Ensure a list is always returned
    return archives if archives else []


@router.get("/{category}")
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
        logger.warning(
            "Invalid limit supplied",
            limit=limit,
            min_limit=MIN_LIMIT
        )
        raise HTTPException(
            status_code=400,
            detail=ERR_LIMIT_POSITIVE,
        )

    try:
        result = replay(category, date, limit)
    except FileNotFoundError as exc:
        logger.error(
            "Archive not found",
            category=category,
            date=date,
            limit=limit,
            error=str(exc),
            exc_info=True
        )
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        logger.error(
            "Invalid arguments for replay",
            category=category,
            date=date,
            limit=limit,
            error=str(exc),
            exc_info=True
        )
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        # Convert unexpected errors to a client‑friendly response
        logger.error(
            "Unexpected error during archive replay",
            category=category,
            date=date,
            limit=limit,
            error=str(exc),
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=ERR_RETRIEVE_ARCHIVE.format(exc=exc),
        ) from exc

    # Ensure the endpoint always returns a list
    return result if result else []