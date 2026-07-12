"""Trade archive replay endpoints."""
from datetime import datetime
from typing import Any, List, Dict

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, validator

from app.api.deps import get_current_user
from app.models.user import User
from app.archive.trade_archiver import replay, list_archives

router = APIRouter(prefix="/archive", tags=["archive"])


class ArchiveItem(BaseModel):
    """Schema representing a single archived trade record."""

    id: str = Field(
        ...,
        description="Unique identifier of the archived trade.",
        example="trade_12345",
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp of the trade.",
        example="2023-09-15T14:30:00Z",
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Arbitrary trade data payload.",
        example={"price": 123.45, "quantity": 100},
    )


class ArchiveListResponse(BaseModel):
    """Response schema for the archive index endpoint."""

    archives: List[str] = Field(
        ...,
        description="Available archive categories.",
        example=["equities", "forex", "options"],
    )


class ArchiveReplayResponse(BaseModel):
    """Response schema for the archive replay endpoint."""

    category: str = Field(
        ...,
        description="Archive category requested.",
        example="equities",
    )
    date: datetime = Field(
        ...,
        description="Date of the archive data.",
        example="2023-09-15T00:00:00Z",
    )
    limit: int = Field(
        ...,
        description="Maximum number of records returned.",
        example=500,
        le=5000,
    )
    records: List[ArchiveItem] = Field(
        ...,
        description="List of archived trade records.",
    )


class ArchiveQueryParams(BaseModel):
    """Validated query parameters for the archive replay endpoint."""

    date: str | None = Field(
        None,
        description="YYYY-MM-DD, defaults to today.",
        example="2023-09-15",
    )
    limit: int = Field(
        500,
        le=5000,
        description="Maximum number of records to return.",
        example=500,
    )

    @validator("date")
    def validate_date(cls, value: str | None) -> str | None:
        """Ensure the provided date matches YYYY‑MM‑DD format."""
        if value is None:
            return value
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("date must be in YYYY-MM-DD format") from exc
        return value


@router.get("/index", response_model=ArchiveListResponse)
async def get_index(current_user: User = Depends(get_current_user)):
    """Return a list of available archive categories."""
    return ArchiveListResponse(archives=list_archives())


@router.get("/{category}", response_model=ArchiveReplayResponse)
async def get_archive(
    category: str,
    date: str | None = Query(None, description="YYYY-MM-DD, defaults to today"),
    limit: int = Query(500, le=5000, description="Maximum number of records to return"),
    current_user: User = Depends(get_current_user),
):
    """Replay archived trades for a given category and date."""
    # Validate query parameters using the Pydantic model
    params = ArchiveQueryParams(date=date, limit=limit)

    raw_records = replay(category, params.date, params.limit)

    # Convert raw records to ArchiveItem instances; assume each record is a mapping with required keys
    records = [
        ArchiveItem(
            id=rec.get("id"),
            timestamp=rec.get("timestamp"),
            data=rec.get("data", {}),
        )
        for rec in raw_records
    ]

    # Determine the effective date for the response
    response_date = (
        datetime.strptime(params.date, "%Y-%m-%d")
        if params.date
        else datetime.utcnow()
    )

    return ArchiveReplayResponse(
        category=category,
        date=response_date,
        limit=params.limit,
        records=records,
    )