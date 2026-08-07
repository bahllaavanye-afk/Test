"""Integrations endpoints: Notion sync, etc."""
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
    if current_user is None:
        raise ValueError("current_user must not be None")
    start_time = time.perf_counter()
    sync = get_notion_sync()
    result = {
        KEY_ENABLED: sync.enabled,
        KEY_NOTION_TOKEN_SET: bool(sync.notion_token),
        KEY_NOTION_DB_ID_SET: bool(sync.notion_db_id),
        KEY_GITHUB_TOKEN_SET: bool(sync.github_token),
        KEY_GITHUB_REPO: sync.github_repo or None,
    }
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Notion status retrieved",
        extra={
            "user_id": current_user.id,
            "signal_count": 0,
            "execution_time_ms": elapsed_ms,
            "pnl": None,
        },
    )
    return result


@router.post(NOTION_SYNC_PATH)
async def trigger_notion_sync(current_user: User = Depends(get_current_user)):
    """Trigger a bidirectional GitHub Issues ↔ Notion sync."""
    if current_user is None:
        raise ValueError("current_user must not be None")
    sync = get_notion_sync()
    if sync is None:
        raise ValueError("sync instance must not be None")
    start_time = time.perf_counter()
    sync_result = await sync.sync_all()
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Attempt to extract metrics from sync_result if available
    signal_count = None
    pnl = None
    if isinstance(sync_result, dict):
        signal_count = sync_result.get("signal_count")
        pnl = sync_result.get("pnl")

    logger.info(
        "Notion sync executed",
        extra={
            "user_id": current_user.id,
            "signal_count": signal_count,
            "execution_time_ms": elapsed_ms,
            "pnl": pnl,
        },
    )
    return sync_result