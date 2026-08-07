"""Trade archive replay endpoints.

Provides FastAPI routes to list available trade archives and replay trades for a
given category, optional date, and limit. The endpoints enforce validation of
input parameters and translate unexpected errors into HTTP exceptions.
"""

from typing import Any, List, Optional

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

router = APIRouter(prefix=ARCHIVE_PREFIX, tags=[ARCHIVE_TAG])


def _normalize_date(date: Optional[str]) -> Optional[str]:
    """Convert empty strings to ``None``; otherwise return the original value.

    Args:
        date: Date string supplied by the client, or ``None``.

    Returns:
        ``None`` if ``date`` is an empty string, otherwise the original ``date``.
    """
    return None if date == "" else date


def _validate_limit(limit: int) -> None:
    """Validate that ``limit`` is within the allowed range.

    Raises:
        HTTPException: If ``limit`` is less than the minimum allowed value.
    """
    if limit < MIN_LIMIT:
        raise HTTPException(status_code=400, detail=ERR_LIMIT_POSITIVE)


def _execute_replay(category: str, date: Optional[str], limit: int) -> List[Any]:
    """Execute the replay function and surface errors as HTTP exceptions.

    Args:
        category: Archive category to replay.
        date: Optional date filter.
        limit: Maximum number of records to return.

    Returns:
        A list of replayed trade records.

    Raises:
        HTTPException: If the underlying ``replay`` call raises an exception.
    """
    try:
        return replay(category, date, limit)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=ERR_RETRIEVE_ARCHIVE.format(exc=exc),
        ) from exc


@router.get("/index")
async def get_index(current_user: User = Depends(get_current_user)) -> List[Any]:
    """Return a list of available archives.

    If the underlying ``list_archives`` function returns ``None``, an empty list
    is returned instead.

    Args:
        current_user: Authenticated user (injected by dependency).

    Returns:
        List of archive identifiers.
    """
    archives = list_archives()
    return archives if archives else []


@router.get("/{category}")
async def get_archive(
    category: str,
    date: Optional[str] = Query(
        None, description=DATE_DESCRIPTION
    ),
    limit: int = Query(
        DEFAULT_LIMIT,
        ge=MIN_LIMIT,
        le=MAX_LIMIT,
        description=LIMIT_DESCRIPTION,
    ),
    current_user: User = Depends(get_current_user),
) -> List[Any]:
    """Replay trades for a given category and optional date.

    Edge‑case handling:
    * ``date`` empty string is treated as ``None``.
    * ``limit`` is validated to be at least ``1``.
    * Returns an empty list if the replay yields no data.

    Args:
        category: Archive category to replay.
        date: Optional date filter; empty strings are normalized to ``None``.
        limit: Maximum number of records to return.
        current_user: Authenticated user (injected by dependency).

    Returns:
        List of replayed trade records, possibly empty.
    """
    normalized_date = _normalize_date(date)
    _validate_limit(limit)
    result = _execute_replay(category, normalized_date, limit)
    return result if result else []