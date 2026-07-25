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


def _validate_user(user: User) -> None:
    """Validate that the provided user object is a proper User instance."""
    if user is None:
        raise ValueError("Current user must be provided.")
    if not isinstance(user, User):
        raise ValueError(f"Invalid user type: expected User, got {type(user).__name__}.")


@router.get(NOTION_STATUS_PATH)
async def notion_status(current_user: User = Depends(get_current_user)):
    """Whether Notion sync is configured."""
    _validate_user(current_user)
    sync = get_notion_sync()
    if sync is None:
        raise ValueError("Notion sync configuration could not be retrieved.")
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
    _validate_user(current_user)
    sync = get_notion_sync()
    if sync is None:
        raise ValueError("Notion sync configuration could not be retrieved.")
    if not sync.enabled:
        raise ValueError("Notion synchronization is disabled; cannot trigger sync.")
    return await sync.sync_all()