"""Integrations endpoints: Notion sync, etc."""
from fastapi import APIRouter, Depends, HTTPException, status
from app.api.deps import get_current_user
from app.integrations.notion_sync import get_notion_sync
from app.models.user import User

# Constants
INTEGRATIONS_PREFIX: str = "/integrations"
INTEGRATIONS_TAG: str = "integrations"

NOTION_STATUS_PATH: str = "/notion/status"
NOTION_SYNC_PATH: str = "/notion/sync"

KEY_ENABLED: str = "enabled"
KEY_NOTION_TOKEN_SET: str = "notion_token_set"
KEY_NOTION_DB_ID_SET: str = "notion_db_id_set"
KEY_GITHUB_TOKEN_SET: str = "github_token_set"
KEY_GITHUB_REPO: str = "github_repo"

router = APIRouter(prefix=INTEGRATIONS_PREFIX, tags=[INTEGRATIONS_TAG])


@router.get(NOTION_STATUS_PATH)
async def notion_status(current_user: User = Depends(get_current_user)) -> dict:
    """Whether Notion sync is configured."""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    sync = get_notion_sync()
    return {
        KEY_ENABLED: sync.enabled,
        KEY_NOTION_TOKEN_SET: bool(sync.notion_token),
        KEY_NOTION_DB_ID_SET: bool(sync.notion_db_id),
        KEY_GITHUB_TOKEN_SET: bool(sync.github_token),
        KEY_GITHUB_REPO: sync.github_repo or None,
    }


@router.post(NOTION_SYNC_PATH)
async def trigger_notion_sync(current_user: User = Depends(get_current_user)):
    """Trigger a bidirectional GitHub Issues ↔ Notion sync."""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    sync = get_notion_sync()
    if sync is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Notion sync instance is unavailable",
        )
    return await sync.sync_all()