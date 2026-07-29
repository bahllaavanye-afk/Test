"""API v1 router — mounts all sub-routers."""
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field, validator

from app.api.v1 import (
    auth,
    accounts,
    orders,
    positions,
    trades,
    strategies,
    backtests,
    comparison,
    experiments,
    ml,
    risk,
    market_data,
    analytics,
    agents,
    notifications,
    archive,
    improvements,
    monitoring,
    integrations,
    pipeline,
    leaderboard,
    releases,
    bots,
)
from app.api.v1.scanners import router as scanners_router
from app.api.v1.options import router as options_router
from app.api.v1.regime import router as regime_router
from app.api.v1.audit_log import router as audit_log_router
from app.api.v1.bots import router as bots_router
from app.api.v1.discord_interactions import router as discord_router
from app.api.v1.webhooks import router as webhooks_router


class HealthResponse(BaseModel):
    """Schema for the health‑check endpoint."""
    status: str = Field(
        ...,
        description="Health status of the API",
        example="healthy",
    )
    timestamp: datetime = Field(
        ...,
        description="Current server timestamp in ISO‑8601 format",
        example="2023-01-01T00:00:00Z",
    )

    @validator("status")
    def status_must_be_healthy(cls, v: str) -> str:
        """Ensure the status field is always 'healthy'."""
        if v != "healthy":
            raise ValueError("status must be 'healthy'")
        return v


api_router = APIRouter()


@api_router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
    description="Simple endpoint to verify that the API service is operational.",
)
def health_check() -> HealthResponse:
    """Return a basic health status with the current UTC timestamp."""
    return HealthResponse(status="healthy", timestamp=datetime.utcnow())


api_router.include_router(auth.router)
api_router.include_router(accounts.router)
api_router.include_router(orders.router)
api_router.include_router(positions.router)
api_router.include_router(trades.router)

api_router.include_router(strategies.router)

api_router.include_router(backtests.router)
api_router.include_router(comparison.router)
api_router.include_router(experiments.router)
api_router.include_router(ml.router)
api_router.include_router(risk.router)
api_router.include_router(market_data.router)
# Underscore-prefix alias so /market_data/* and /market-data/* both resolve
api_router.include_router(market_data.router_underscore)
api_router.include_router(analytics.router)
api_router.include_router(agents.router)
api_router.include_router(notifications.router)
api_router.include_router(archive.router)
api_router.include_router(improvements.router)
api_router.include_router(monitoring.router)
api_router.include_router(options_router)
api_router.include_router(regime_router)
api_router.include_router(audit_log_router)
api_router.include_router(integrations.router)
api_router.include_router(pipeline.router)
api_router.include_router(leaderboard.router)
api_router.include_router(releases.router)
api_router.include_router(bots_router)
api_router.include_router(scanners_router)
api_router.include_router(discord_router)
api_router.include_router(webhooks_router)