"""API v1 router — mounts all sub-routers."""
from fastapi import APIRouter
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any

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


class ErrorResponse(BaseModel):
    """Standard error response model for API endpoints."""
    code: int = Field(
        ...,
        description="HTTP status code of the error.",
        example=400,
    )
    message: str = Field(
        ...,
        description="Human‑readable error message.",
        example="Invalid request payload.",
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context about the error, such as validation issues.",
        example={"field": "value", "error": "must be positive"},
    )

    @validator("code")
    def code_must_be_valid(cls, v: int) -> int:
        """Ensure the HTTP status code is within the standard range."""
        if not (100 <= v <= 599):
            raise ValueError("code must be a valid HTTP status code between 100 and 599")
        return v


api_router = APIRouter()

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