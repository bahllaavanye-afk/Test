"""Integrations endpoints: Notion sync, etc."""
import logging

from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_current_user
from app.integrations.notion_sync import get_notion_sync
from app.models.user import User

# Logger configuration
logger = logging.getLogger(__name__)

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
async def notion_status(current_user: User = Depends(get_current_user)):
    """Whether Notion sync is configured."""
    try:
        sync = get_notion_sync()
        return {
            KEY_ENABLED: sync.enabled,
            KEY_NOTION_TOKEN_SET: bool(sync.notion_token),
            KEY_NOTION_DB_ID_SET: bool(sync.notion_db_id),
            KEY_GITHUB_TOKEN_SET: bool(sync.github_token),
            KEY_GITHUB_REPO: sync.github_repo or None,
        }
    except Exception as exc:
        logger.exception("Error retrieving Notion status for user %s", current_user.id)
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve Notion integration status."
        ) from exc


@router.post(NOTION_SYNC_PATH)
async def trigger_notion_sync(current_user: User = Depends(get_current_user)):
    """Trigger a bidirectional GitHub Issues ↔ Notion sync."""
    try:
        sync = get_notion_sync()
        return await sync.sync_all()
    except Exception as exc:
        logger.exception("Error triggering Notion sync for user %s", current_user.id)
        raise HTTPException(
            status_code=500,
            detail="Failed to initiate Notion synchronization."
        ) from exc