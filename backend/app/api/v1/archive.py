"""Trade archive replay endpoints.

Provides HTTP endpoints for listing available trade archives and replaying
trades from a specific archive category. The endpoints rely on the
`list_archives` and `replay` functions from :pymod:`app.archive.trade_archiver`
and enforce basic validation and error handling.
"""

from typing import List, Any, Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from app.api.deps import get_current_user
from app.models.user import User
from app.archive.trade_archiver import replay, list_archives

router = APIRouter(prefix="/archive", tags=["archive"])


@router.get("/index")
async def get_index(current_user: User = Depends(get_current_user)) -> List[Any]:
    """
    Retrieve a list of available archive identifiers.

    Parameters
    ----------
    current_user: User
        The authenticated user, injected by FastAPI's dependency system.

    Returns
    -------
    List[Any]
        A list of archive identifiers. Returns an empty list if no archives
        are available.
    """
    archives = list_archives()
    # Ensure a list is always returned
    return archives if archives else []


@router.get("/{category}")
async def get_archive(
    category: str,
    date: Optional[str] = Query(
        None, description="YYYY-MM-DD, defaults to today"
    ),
    limit: int = Query(
        500,
        ge=1,
        le=5000,
        description="Maximum number of records to return (1-5000)",
    ),
    current_user: User = Depends(get_current_user),
) -> List[Any]:
    """
    Replay trades for a given archive category.

    Parameters
    ----------
    category: str
        The archive category to replay.
    date: Optional[str]
        Target date in ``YYYY-MM-DD`` format. If omitted or an empty string,
        the most recent data is used.
    limit: int
        Upper bound on the number of trade records to return. Must be between
        1 and 5,000 inclusive.
    current_user: User
        The authenticated user, injected by FastAPI's dependency system.

    Returns
    -------
    List[Any]
        A list of replayed trade records. Returns an empty list if the replay
        yields no data.

    Raises
    ------
    HTTPException
        * 400 – If ``limit`` is less than 1 (defensive check).
        * 500 – If an unexpected error occurs while retrieving the archive.
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