"""Integrations endpoints: Notion sync status and trigger.

This module defines FastAPI routes for querying the status of the Notion
integration and for manually triggering a bi‑directional sync between
GitHub Issues and Notion pages. The endpoints are protected by the
standard user authentication dependency.
"""

from typing import Any, Dict

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


@router.get(NOTION_STATUS_PATH)
async def notion_status(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Return a dictionary indicating the configuration status of the Notion integration.

    Args:
        current_user: The authenticated user obtained via dependency injection.

    Returns:
        A mapping with keys:
        - ``enabled``: Whether the Notion sync feature is enabled.
        - ``notion_token_set``: Boolean indicating if a Notion token is stored.
        - ``notion_db_id_set``: Boolean indicating if a Notion database ID is stored.
        - ``github_token_set``: Boolean indicating if a GitHub token is stored.
        - ``github_repo``: The configured GitHub repository name, or ``None`` if not set.
    """
    sync = get_notion_sync()
    return {
        KEY_ENABLED: sync.enabled,
        KEY_NOTION_TOKEN_SET: bool(sync.notion_token),
        KEY_NOTION_DB_ID_SET: bool(sync.notion_db_id),
        KEY_GITHUB_TOKEN_SET: bool(sync.github_token),
        KEY_GITHUB_REPO: sync.github_repo or None,
    }


@router.post(NOTION_SYNC_PATH)
async def trigger_notion_sync(current_user: User = Depends(get_current_user)) -> Any:
    """Manually trigger a full bi‑directional sync between GitHub Issues and Notion.

    Args:
        current_user: The authenticated user obtained via dependency injection.

    Returns:
        The result of ``sync_all`` from the Notion sync service, which may be
        a status dictionary or any other payload defined by the integration logic.
    """
    sync = get_notion_sync()
    return await sync.sync_all()