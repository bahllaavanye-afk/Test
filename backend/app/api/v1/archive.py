"""Trade archive replay endpoints."""
from datetime import date as dt_date, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field, validator

from app.api.deps import get_current_user
from app.models.user import User
from app.archive.trade_archiver import replay, list_archives

router = APIRouter(prefix="/archive", tags=["archive"])


class ArchiveInfo(BaseModel):
    """Metadata about a stored trade archive."""

    category: str = Field(
        ...,
        description="The trade category the archive belongs to.",
        example="equities",
    )
    date: dt_date = Field(
        ...,
        description="Date of the archive in ISO format (YYYY‑MM‑DD).",
        example="2023-08-15",
    )
    record_count: int = Field(
        ...,
        ge=0,
        description="Number of trade records stored in the archive.",
        example=1245,
    )


class TradeRecord(BaseModel):
    """A single trade entry returned from an archive replay."""

    trade_id: str = Field(
        ...,
        description="Unique identifier for the trade.",
        example="TRD-20230815-0001",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol of the traded instrument.",
        example="AAPL",
    )
    quantity: float = Field(
        ...,
        gt=0,
        description="Number of units traded; must be positive.",
        example=150.0,
    )
    price: float = Field(
        ...,
        gt=0,
        description="Execution price per unit; must be positive.",
        example=172.35,
    )
    timestamp: datetime = Field(
        ...,
        description="Exact time the trade was executed (UTC).",
        example="2023-08-15T14:32:10Z",
    )

    @validator("timestamp")
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone‑aware; assume UTC if naive."""
        if v.tzinfo is None:
            return v.replace(tzinfo=datetime.timezone.utc)
        return v


@router.get("/index", response_model=List[ArchiveInfo])
async def get_index(current_user: User = Depends(get_current_user)):
    """
    Return a list of available archives.
    Handles the case where the underlying function returns None.
    """
    archives = list_archives()
    # Ensure a list is always returned
    return archives if archives else []


@router.get(
    "/{category}",
    response_model=List[TradeRecord],
)
async def get_archive(
    category: str,
    date: Optional[str] = Query(
        None,
        description="YYYY-MM-DD, defaults to today",
        example="2023-08-15",
    ),
    limit: int = Query(
        500,
        ge=1,
        le=5000,
        description="Maximum number of records to return (1‑5000)",
        example=250,
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