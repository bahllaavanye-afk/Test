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
    return await sync.sync_all()


# ---------------------------------------------------------------------------
# Unit tests for edge cases
# ---------------------------------------------------------------------------
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_sync():
    """Create a generic mock sync object with default attributes."""
    sync = MagicMock()
    sync.enabled = True
    sync.notion_token = "dummy-token"
    sync.notion_db_id = "dummy-db"
    sync.github_token = "gh-token"
    sync.github_repo = "repo/name"
    sync.sync_all = AsyncMock(return_value={"result": "ok"})
    return sync


@pytest.mark.asyncio
async def test_notion_status_all_fields_none(mock_sync):
    """Edge case: all optional fields are None or empty strings."""
    # Configure mock to have falsy token values
    mock_sync.enabled = False
    mock_sync.notion_token = ""
    mock_sync.notion_db_id = None
    mock_sync.github_token = ""
    mock_sync.github_repo = ""

    with patch("app.integrations.notion_sync.get_notion_sync", return_value=mock_sync):
        # Directly call the endpoint function
        result = await notion_status()
        assert result[KEY_ENABLED] is False
        assert result[KEY_NOTION_TOKEN_SET] is False
        assert result[KEY_NOTION_DB_ID_SET] is False
        assert result[KEY_GITHUB_TOKEN_SET] is False
        # github_repo should be None when the value is falsy
        assert result[KEY_GITHUB_REPO] is None


@pytest.mark.asyncio
async def test_notion_status_non_boolean_enabled(mock_sync):
    """Edge case: enabled attribute is not a bool (e.g., None)."""
    mock_sync.enabled = None  # non‑boolean truthy/falsy value
    mock_sync.notion_token = "t"
    mock_sync.notion_db_id = "d"
    mock_sync.github_token = "g"
    mock_sync.github_repo = "repo"

    with patch("app.integrations.notion_sync.get_notion_sync", return_value=mock_sync):
        result = await notion_status()
        # The endpoint should return the raw value without coercion
        assert result[KEY_ENABLED] is None
        assert result[KEY_NOTION_TOKEN_SET] is True
        assert result[KEY_NOTION_DB_ID_SET] is True
        assert result[KEY_GITHUB_TOKEN_SET] is True
        assert result[KEY_GITHUB_REPO] == "repo"


@pytest.mark.asyncio
async def test_trigger_notion_sync_propagates_exception(mock_sync):
    """Edge case: sync_all raises an exception; ensure it propagates."""
    async def raise_error():
        raise RuntimeError("sync failure")

    mock_sync.sync_all = AsyncMock(side_effect=raise_error)

    with patch("app.integrations.notion_sync.get_notion_sync", return_value=mock_sync):
        with pytest.raises(RuntimeError, match="sync failure"):
            await trigger_notion_sync()