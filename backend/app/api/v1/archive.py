"""Trade archive replay endpoints."""
from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field, validator

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


class ArchiveQueryParams(BaseModel):
    """Query parameters for archive retrieval."""

    date: Optional[date] = Field(
        default=None,
        description=DATE_DESCRIPTION,
        example="2023-09-15",
    )
    limit: int = Field(
        default=DEFAULT_LIMIT,
        ge=MIN_LIMIT,
        le=MAX_LIMIT,
        description=LIMIT_DESCRIPTION,
        example=100,
    )

    @validator("date", pre=True)
    def empty_string_to_none(cls, v):
        """Treat empty string as None for optional date."""
        if v == "":
            return None
        return v


class Trade(BaseModel):
    """Schema representing a single trade record."""

    trade_id: str = Field(..., description="Unique identifier for the trade")
    timestamp: datetime = Field(..., description="Timestamp of the trade execution")
    symbol: str = Field(..., description="Ticker symbol")
    side: str = Field(
        ...,
        description="Side of the trade, either 'buy' or 'sell'",
        example="buy",
    )
    quantity: float = Field(..., description="Number of shares or contracts")
    price: float = Field(..., description="Execution price per unit")

    @validator("side")
    def validate_side(cls, v):
        allowed = {"buy", "sell"}
        if v.lower() not in allowed:
            raise ValueError(f"side must be one of {allowed}")
        return v.lower()


@router.get(
    "/index",
    response_model=List[str],
    summary="List available archives",
    description="Return a list of available archives. Ensures an empty list is returned if none exist.",
)
async def get_index(current_user: User = Depends(get_current_user)):
    archives = list_archives()
    return archives if archives else []


@router.get(
    "/{category}",
    response_model=List[Trade],
    summary="Replay trades from an archive",
    description=(
        "Replay trades for a given category and optional date. "
        "Empty date strings are treated as None. "
        "Limit is validated to be within allowed bounds. "
        "Returns an empty list if no trades are found."
    ),
)
async def get_archive(
    category: str = Path(..., description="Archive category", example="equities"),
    params: ArchiveQueryParams = Depends(),
    current_user: User = Depends(get_current_user),
):
    if params.limit < MIN_LIMIT:
        raise HTTPException(status_code=400, detail=ERR_LIMIT_POSITIVE)

    try:
        result = replay(category, params.date, params.limit)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=ERR_RETRIEVE_ARCHIVE.format(exc=exc),
        ) from exc

    return result if result else []