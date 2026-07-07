"""Integrations endpoints: Notion sync, Slack test, etc."""
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.integrations.notion_sync import get_notion_sync
from app.models.user import User

# Constants
INTEGRATIONS_PREFIX = "/integrations"
INTEGRATIONS_TAG = "integrations"

ENDPOINT_NOTION_STATUS = "/notion/status"
ENDPOINT_NOTION_SYNC = "/notion/sync"

KEY_ENABLED = "enabled"
KEY_NOTION_TOKEN_SET = "notion_token_set"
KEY_NOTION_DB_ID_SET = "notion_db_id_set"
KEY_GITHUB_TOKEN_SET = "github_token_set"
KEY_GITHUB_REPO = "github_repo"

router = APIRouter(prefix=INTEGRATIONS_PREFIX, tags=[INTEGRATIONS_TAG])


@router.get(ENDPOINT_NOTION_STATUS)
async def notion_status(current_user: User = Depends(get_current_user)):
    """Whether Notion sync is configured."""
    sync = get_notion_sync()
    return {
        KEY_ENABLED: sync.enabled,
        KEY_NOTION_TOKEN_SET: bool(sync.notion_token),
        KEY_NOTION_DB_ID_SET: bool(sync.notion_db_id),
        KEY_GITHUB_TOKEN_SET: bool(sync.github_token),
        KEY_GITHUB_REPO: sync.github_repo or None,
    }


@router.post(ENDPOINT_NOTION_SYNC)
async def trigger_notion_sync(current_user: User = Depends(get_current_user)):
    """Trigger a bidirectional GitHub Issues ↔ Notion sync."""
    sync = get_notion_sync()
    return await sync.sync_all()