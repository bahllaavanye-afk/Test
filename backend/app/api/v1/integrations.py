"""Integrations endpoints: Notion sync, Slack test, etc."""
import logging
import time
from fastapi import APIRouter, Depends
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


@router.get(NOTION_STATUS_PATH)
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


@router.post(NOTION_SYNC_PATH)
async def trigger_notion_sync(current_user: User = Depends(get_current_user)):
    """Trigger a bidirectional GitHub Issues ↔ Notion sync."""
    sync = get_notion_sync()
    start_time = time.perf_counter()
    result = await sync.sync_all()
    elapsed = time.perf_counter() - start_time

    # Determine signal count
    try:
        signal_count = len(result)  # type: ignore[arg-type]
    except Exception:
        signal_count = None

    # Extract P&L if present
    pnl = None
    if isinstance(result, dict):
        pnl = result.get("pnl")
    elif hasattr(result, "pnl"):
        pnl = getattr(result, "pnl")

    logger.info(
        "Notion sync completed",
        extra={
            "signal_count": signal_count,
            "execution_time_seconds": elapsed,
            "pnl": pnl,
        },
    )
    return result