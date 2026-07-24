"""Integrations endpoints: Notion sync, Slack test, etc."""
from fastapi import APIRouter, Depends, HTTPException
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
    """Validate the injected user object."""
    if user is None:
        raise ValueError("User dependency returned None")
    if not isinstance(user, User):
        raise ValueError(f"Expected User instance, got {type(user)}")
    # Additional user-specific validation can be added here


def _validate_sync(sync) -> None:
    """Validate the Notion sync object."""
    if sync is None:
        raise ValueError("Notion sync instance is not available")
    # Ensure required attributes exist
    required_attrs = ["enabled", "notion_token", "notion_db_id", "github_token", "github_repo"]
    missing = [attr for attr in required_attrs if not hasattr(sync, attr)]
    if missing:
        raise ValueError(f"Notion sync object missing attributes: {', '.join(missing)}")


@router.get(NOTION_STATUS_PATH)
async def notion_status(current_user: User = Depends(get_current_user)):
    """Whether Notion sync is configured."""
    try:
        _validate_user(current_user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    sync = get_notion_sync()
    try:
        _validate_sync(sync)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

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
    try:
        _validate_user(current_user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    sync = get_notion_sync()
    try:
        _validate_sync(sync)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return await sync.sync_all()