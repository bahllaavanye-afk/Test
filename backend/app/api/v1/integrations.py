"""Integrations endpoints: Notion sync, Slack test, etc."""
from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.integrations.notion_sync import get_notion_sync
from app.models.user import User

router = APIRouter(prefix="/integrations", tags=["integrations"])


@router.get("/notion/status")
async def notion_status(current_user: User = Depends(get_current_user)):
    """Whether Notion sync is configured."""
    sync = get_notion_sync()
    return {
        "enabled": sync.enabled,
        "notion_token_set": bool(sync.notion_token),
        "notion_db_id_set": bool(sync.notion_db_id),
        "github_token_set": bool(sync.github_token),
        "github_repo": sync.github_repo or None,
    }


@router.post("/notion/sync")
async def trigger_notion_sync(current_user: User = Depends(get_current_user)):
    """Trigger a bidirectional GitHub Issues ↔ Notion sync."""
    sync = get_notion_sync()
    return await sync.sync_all()
