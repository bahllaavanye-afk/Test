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

router = APIRouter(prefix=ARCHIVE_PREFIX, tags=[ARCHIVE_TAG])


def _normalize_date(date: str | None) -> str | None:
    """Convert empty strings to ``None``; leave other values unchanged."""
    return None if date == "" else date


def _validate_limit(limit: int) -> None:
    """Raise an ``HTTPException`` if ``limit`` is outside the allowed range."""
    if limit < MIN_LIMIT:
        raise HTTPException(status_code=400, detail=ERR_LIMIT_POSITIVE)


def _run_replay(category: str, date: str | None, limit: int) -> list:
    """
    Execute ``replay`` and wrap unexpected exceptions in an HTTPException.

    Returns:
        list: The replay result (may be empty).
    """
    try:
        return replay(category, date, limit)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail=ERR_RETRIEVE_ARCHIVE.format(exc=exc),
        ) from exc


def _prepare_and_execute(category: str, date: str | None, limit: int) -> list:
    """
    Consolidate parameter preparation and replay execution for the archive endpoint.

    This helper isolates the core logic of ``get_archive`` to improve readability.
    """
    normalized_date = _normalize_date(date)
    _validate_limit(limit)
    return _run_replay(category, normalized_date, limit)


@router.get("/index")
async def get_index(current_user: User = Depends(get_current_user)):
    """
    Return a list of available archives.

    Guarantees that a list (possibly empty) is always returned.
    """
    archives = list_archives()
    return archives if archives else []


@router.get("/{category}")
async def get_archive(
    category: str,
    date: str | None = Query(
        None,
        description=DATE_DESCRIPTION,
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
    Replay trades for a given ``category`` and optional ``date``.

    * Empty ``date`` strings are treated as ``None``.
    * ``limit`` must be at least ``1``.
    * Returns an empty list when the replay yields no data.
    """
    result = _prepare_and_execute(category, date, limit)
    return result if result else []