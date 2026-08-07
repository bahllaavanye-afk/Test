"""Integrations endpoints: Notion sync, etc."""
import logging
from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_current_user
from app.integrations.notion_sync import get_notion_sync
from app.models.user import User

# Logger setup
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


def _ensure_user(current_user: User | None) -> User:
    """Validate that a user object is provided."""
    if current_user is None:
        raise ValueError("current_user must not be None")
    return current_user


def _get_sync_instance() -> object:
    """Retrieve the Notion sync instance, ensuring it exists."""
    sync = get_notion_sync()
    if sync is None:
        raise ValueError("sync instance must not be None")
    return sync


@router.get(NOTION_STATUS_PATH)
async def notion_status(current_user: User = Depends(get_current_user)):
    """Whether Notion sync is configured."""
    try:
        _ensure_user(current_user)
        sync = get_notion_sync()
        return {
            KEY_ENABLED: sync.enabled,
            KEY_NOTION_TOKEN_SET: bool(sync.notion_token),
            KEY_NOTION_DB_ID_SET: bool(sync.notion_db_id),
            KEY_GITHUB_TOKEN_SET: bool(sync.github_token),
            KEY_GITHUB_REPO: sync.github_repo or None,
        }
    except ValueError as ve:
        logger.error("Invalid request in notion_status: %s", ve, exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error("Unexpected error in notion_status", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(NOTION_SYNC_PATH)
async def trigger_notion_sync(current_user: User = Depends(get_current_user)):
    """Trigger a bidirectional GitHub Issues ↔ Notion sync."""
    try:
        _ensure_user(current_user)
        sync = _get_sync_instance()
        return await sync.sync_all()
    except ValueError as ve:
        logger.error("Invalid request in trigger_notion_sync: %s", ve, exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error("Unexpected error in trigger_notion_sync", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")