"""Notifications and activity tracker endpoints."""
from fastapi import APIRouter, Depends, Query
from app.api.deps import get_current_user
from app.models.user import User
from app.notifications.tracker import tracker
from app.notifications.slack import slack

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("/activity")
async def get_activity(
    limit: int = Query(100, le=500),
    category: str | None = None,
    current_user: User = Depends(get_current_user),
):
    return tracker.recent(limit=limit, category=category)


@router.get("/stats")
async def get_stats(current_user: User = Depends(get_current_user)):
    return tracker.stats()


@router.post("/slack/test")
async def slack_test(current_user: User = Depends(get_current_user)):
    """Send a test message to confirm Slack webhook is configured."""
    ok = await slack.notify_system("QuantEdge Slack notifications are working ✓", level="info")
    return {"sent": ok, "enabled": slack._enabled}
