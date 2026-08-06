"""Integrations endpoints: Notion sync, etc."""
from fastapi import APIRouter, Depends
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


def _ensure_user(current_user: User | None) -> User:
    """Validate that a user instance is provided."""
    if current_user is None:
        raise ValueError("current_user must not be None")
    return current_user


def _get_sync() -> "NotionSync":
    """Retrieve the Notion sync instance, ensuring it exists."""
    sync = get_notion_sync()
    if sync is None:
        raise ValueError("sync instance must not be None")
    return sync


def _build_status_response(sync: "NotionSync") -> dict:
    """Construct the status payload for the Notion integration."""
    return {
        KEY_ENABLED: sync.enabled,
        KEY_NOTION_TOKEN_SET: bool(sync.notion_token),
        KEY_NOTION_DB_ID_SET: bool(sync.notion_db_id),
        KEY_GITHUB_TOKEN_SET: bool(sync.github_token),
        KEY_GITHUB_REPO: sync.github_repo or None,
    }


@router.get(NOTION_STATUS_PATH)
async def notion_status(current_user: User = Depends(get_current_user)):
    """Whether Notion sync is configured."""
    _ensure_user(current_user)
    sync = get_notion_sync()
    return _build_status_response(sync)


@router.post(NOTION_SYNC_PATH)
async def trigger_notion_sync(current_user: User = Depends(get_current_user)):
    """Trigger a bidirectional GitHub Issues ↔ Notion sync."""
    _ensure_user(current_user)
    sync = _get_sync()
    return await sync.sync_all()