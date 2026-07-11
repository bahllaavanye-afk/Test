"""Trade archive replay endpoints."""

# Constants
ROUTER_PREFIX = "/archive"
ROUTER_TAGS = ["archive"]
ENDPOINT_INDEX = "/index"
ENDPOINT_CATEGORY = "/{category}"
DATE_DESCRIPTION = "YYYY-MM-DD, defaults to today"
DEFAULT_LIMIT = 500
MAX_LIMIT = 5000

from fastapi import APIRouter, Depends, Query
from app.api.deps import get_current_user
from app.models.user import User
from app.archive.trade_archiver import replay, list_archives

router = APIRouter(prefix=ROUTER_PREFIX, tags=ROUTER_TAGS)


@router.get(ENDPOINT_INDEX)
async def get_index(current_user: User = Depends(get_current_user)):
    return list_archives()


@router.get(ENDPOINT_CATEGORY)
async def get_archive(
    category: str,
    date: str | None = Query(None, description=DATE_DESCRIPTION),
    limit: int = Query(DEFAULT_LIMIT, le=MAX_LIMIT),
    current_user: User = Depends(get_current_user),
):
    return replay(category, date, limit)