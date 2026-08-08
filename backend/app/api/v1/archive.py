"""Trade archive replay endpoints."""
import logging
import time
from typing import Any, List

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

# Endpoint path constants
ROUTE_INDEX: str = "/index"
ROUTE_CATEGORY: str = "/{category}"
EMPTY_STRING: str = ""

router = APIRouter(prefix=ARCHIVE_PREFIX, tags=[ARCHIVE_TAG])


def _normalize_date(date: str | None) -> str | None:
    """Convert empty strings to None; leave other values unchanged."""
    return None if date == EMPTY_STRING else date


def _validate_limit(limit: int) -> None:
    """Raise an HTTPException if limit is not within allowed bounds."""
    if limit < MIN_LIMIT:
        raise HTTPException(status_code=400, detail=ERR_LIMIT_POSITIVE)


def _execute_replay(category: str, date: str | None, limit: int) -> List[Any]:
    """Run the replay function and translate unexpected errors to HTTPException."""
    try:
        return replay(category, date, limit)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=ERR_RETRIEVE_ARCHIVE.format(exc=exc),
        ) from exc


@router.get(ROUTE_INDEX)
async def get_index(current_user: User = Depends(get_current_user)):
    """
    Return a list of available archives.
    Handles the case where the underlying function returns None.
    """
    archives = list_archives()
    # Ensure a list is always returned
    result = archives if archives else []
    logger.info(
        "Archive index retrieved",
        extra={
            "user_id": getattr(current_user, "id", None),
            "archive_count": len(result),
        },
    )
    return result


@router.get(ROUTE_CATEGORY)
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

    start_time = time.time()
    result = _execute_replay(category, normalized_date, limit)
    elapsed = time.time() - start_time

    # Compute metrics
    signal_count = len(result) if result else 0
    total_pnl: float | None = None
    if result and isinstance(result, list):
        pnl_values = [
            trade.get("pnl")
            for trade in result
            if isinstance(trade, dict) and "pnl" in trade and isinstance(trade["pnl"], (int, float))
        ]
        total_pnl = sum(pnl_values) if pnl_values else 0.0

    logger.info(
        "Archive replay executed",
        extra={
            "user_id": getattr(current_user, "id", None),
            "category": category,
            "date": normalized_date,
            "limit": limit,
            "signal_count": signal_count,
            "execution_time_seconds": elapsed,
            "total_pnl": total_pnl,
        },
    )

    # Ensure the endpoint always returns a list
    return result if result else []