"""Integrations endpoints: Notion sync, Slack test, etc."""
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, root_validator

from app.api.deps import get_current_user
from app.integrations.notion_sync import get_notion_sync
from app.models.user import User

router = APIRouter(prefix="/integrations", tags=["integrations"])


class NotionStatusResponse(BaseModel):
    enabled: bool = Field(
        ..., description="Indicates if Notion synchronization is enabled.", example=True
    )
    notion_token_set: bool = Field(
        ...,
        description="True if a Notion integration token has been configured.",
        example=True,
    )
    notion_db_id_set: bool = Field(
        ...,
        description="True if a Notion database ID has been configured.",
        example=False,
    )
    github_token_set: bool = Field(
        ...,
        description="True if a GitHub token for issue synchronization is present.",
        example=True,
    )
    github_repo: Optional[str] = Field(
        None,
        description="The GitHub repository identifier (owner/repo) used for sync, if any.",
        example="quantedge/trading-platform",
    )

    @root_validator
    def check_github_repo_consistency(cls, values):
        github_token_set = values.get("github_token_set")
        github_repo = values.get("github_repo")
        if github_token_set and not github_repo:
            raise ValueError(
                "github_repo must be provided when github_token_set is True."
            )
        return values


class NotionSyncResponse(BaseModel):
    result: Any = Field(
        ...,
        description="Result payload returned from the Notion sync operation.",
        example={"synced_issues": 12, "updated_pages": 5},
    )


@router.get("/notion/status", response_model=NotionStatusResponse)
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


@router.post("/notion/sync", response_model=NotionSyncResponse)
async def trigger_notion_sync(current_user: User = Depends(get_current_user)):
    """Trigger a bidirectional GitHub Issues ↔ Notion sync."""
    sync = get_notion_sync()
    result = await sync.sync_all()
    return {"result": result}