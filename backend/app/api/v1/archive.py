"""Trade archive replay endpoints."""
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Any

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
CONFIDENCE_DESCRIPTION: str = "Minimum confidence (0.0‑1.0) for trade signals"

ERR_LIMIT_POSITIVE: str = "Limit must be a positive integer."
ERR_RETRIEVE_ARCHIVE: str = "Failed to retrieve archive: {exc}"
ERR_CATEGORY_INVALID: str = "Invalid category. Allowed categories: {allowed}"
ERR_CONFIDENCE_RANGE: str = "Confidence must be between 0.0 and 1.0."

router = APIRouter(prefix=ARCHIVE_PREFIX, tags=[ARCHIVE_TAG])

# Allowed categories – tighten entry conditions by restricting to known sets
ALLOWED_CATEGORIES = {"equities", "futures", "forex", "options"}

def _filter_trades(trades: List[Any], min_confidence: float) -> List[Any]:
    """
    Apply confirmation filters to raw trade data.

    A trade is kept if:
    * It contains a ``confidence`` field and the value meets ``min_confidence``.
    * It has a truthy ``valid`` flag (if present).
    * It includes an ``exit_time`` (ensuring proper exit logic).

    This function is defensive – any trade missing the expected fields is discarded.
    """
    filtered: List[Any] = []
    for trade in trades:
        # Expect trade to be a mapping; skip if not
        if not isinstance(trade, dict):
            continue
        confidence = trade.get("confidence")
        if confidence is None or not isinstance(confidence, (int, float)):
            continue
        if confidence < min_confidence:
            continue
        if trade.get("valid", True) is False:
            continue
        if trade.get("exit_time") is None:
            continue
        filtered.append(trade)
    return filtered

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
        None, description=DATE_DESCRIPTION
    ),
    limit: int = Query(
        DEFAULT_LIMIT,
        ge=MIN_LIMIT,
        le=MAX_LIMIT,
        description=LIMIT_DESCRIPTION,
    ),
    min_confidence: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description=CONFIDENCE_DESCRIPTION,
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Replay trades for a given category and optional date.

    Entry tightening:
    * ``category`` must belong to a predefined whitelist.
    * ``min_confidence`` filters out low‑confidence signals.

    Exit integrity:
    * Trades lacking an ``exit_time`` are excluded.

    Edge‑case handling:
    * ``date`` empty string is treated as ``None``.
    * ``limit`` is validated to be at least 1.
    * Returns an empty list if the replay yields no data.
    """
    # Validate category against whitelist
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=ERR_CATEGORY_INVALID.format(allowed=", ".join(sorted(ALLOWED_CATEGORIES))),
        )

    # Normalize empty date strings
    if date == "":
        date = None

    # Defensive check for limit (should already be enforced by Query)
    if limit < MIN_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=ERR_LIMIT_POSITIVE,
        )

    # Validate confidence range (again, enforced by Query, but defensive)
    if not (0.0 <= min_confidence <= 1.0):
        raise HTTPException(
            status_code=400,
            detail=ERR_CONFIDENCE_RANGE,
        )

    try:
        raw_result = replay(category, date, limit)
    except Exception as exc:
        # Convert unexpected errors to a client‑friendly response
        raise HTTPException(
            status_code=500,
            detail=ERR_RETRIEVE_ARCHIVE.format(exc=exc),
        ) from exc

    # Ensure we have a list to work with
    trades: List[Any] = raw_result if isinstance(raw_result, list) else []

    # Apply confirmation filters to tighten signal quality
    filtered_trades = _filter_trades(trades, min_confidence)

    # Return filtered list (empty if none pass)
    return filtered_trades if filtered_trades else []