"""System status endpoint."""
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/system", tags=["system"])

@router.get("/scheduler-status")
async def get_scheduler_status(current_user: User = Depends(get_current_user)):
    """Returns scheduler job status for monitoring."""
    from app.tasks.scheduler import get_scheduler_jobs
    return {
        "jobs": get_scheduler_jobs(),
        "event_chain_enabled": True,
        "order_sync_interval_seconds": 15,
        "strategy_runner_interval_seconds": 10,
        "price_feed_interval_seconds": 2,
    }
