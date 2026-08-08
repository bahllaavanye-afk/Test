"""Trade archive replay endpoints."""
import logging
from fastapi import APIRouter, Depends, Query, HTTPException
from app.api.deps import get_current_user
from app.models.user import User
from app.archive.trade_archiver import replay, list_archives

# Logger configuration
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


def _normalize_date(date: str | None) -> str | None:
    """Convert empty strings to None; leave other values unchanged."""
    return None if date == "" else date


def _validate_limit(limit: int) -> None:
    """Raise an HTTPException if limit is not within allowed bounds."""
    if limit < MIN_LIMIT:
        raise HTTPException(status_code=400, detail=ERR_LIMIT_POSITIVE)


def _execute_replay(category: str, date: str | None, limit: int) -> list:
    """
    Run the replay function and translate errors to HTTPException.
    Specific exceptions are mapped to appropriate HTTP status codes.
    """
    try:
        return replay(category, date, limit)
    except ValueError as exc:
        logger.error(
            "Invalid replay parameters",
            exc_info=exc,
            extra={"category": category, "date": date, "limit": limit},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error(
            "Runtime error during replay",
            exc_info=exc,
            extra={"category": category, "date": date, "limit": limit},
        )
        raise HTTPException(status_code=500, detail=ERR_RETRIEVE_ARCHIVE.format(exc=exc)) from exc
    except Exception as exc:
        logger.error(
            "Unexpected error during replay",
            exc_info=exc,
            extra={"category": category, "date": date, "limit": limit},
        )
        raise HTTPException(status_code=500, detail=ERR_RETRIEVE_ARCHIVE.format(exc=exc)) from exc


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
            "Failed to retrieve archive list",
            exc_info=exc,
            extra={"user_id": getattr(current_user, "id", None)},
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
    normalized_date = _normalize_date(date)
    _validate_limit(limit)
    result = _execute_replay(category, normalized_date, limit)
    # Ensure the endpoint always returns a list
    return result if result else []