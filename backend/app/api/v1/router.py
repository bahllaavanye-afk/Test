"""API v1 router — mounts all sub-routers.

Provides shared Pydantic schemas used across the API for request validation,
response documentation, and automatic OpenAPI generation.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field, validator
from typing import Optional

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
    options,
    regime,
    audit_log,
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


# --------------------------------------------------------------------------- #
# Shared Pydantic Schemas
# --------------------------------------------------------------------------- #

class PaginationParams(BaseModel):
    """Parameters for paginated endpoints."""
    page: int = Field(
        1,
        ge=1,
        description="Page number, starting at 1.",
        example=1,
    )
    size: int = Field(
        20,
        ge=1,
        le=1000,
        description="Number of items per page.",
        example=50,
    )


class DateRange(BaseModel):
    """ISO‑8601 date range used for filtering time‑based data."""
    start_date: str = Field(
        ...,
        description="Start date in ISO‑8601 format (YYYY‑MM‑DD).",
        example="2023-01-01",
    )
    end_date: str = Field(
        ...,
        description="End date in ISO‑8601 format (YYYY‑MM‑DD).",
        example="2023-01-31",
    )

    @validator("end_date")
    def _validate_chronology(cls, v: str, values: dict) -> str:
        """Ensure the end date is not earlier than the start date."""
        start = values.get("start_date")
        if start and v < start:
            raise ValueError("end_date must be on or after start_date")
        return v


class OrderCreate(BaseModel):
    """Schema for creating a new order."""
    symbol: str = Field(
        ...,
        description="Ticker symbol of the asset to trade.",
        example="AAPL",
    )
    quantity: int = Field(
        ...,
        gt=0,
        description="Number of shares/contracts to trade.",
        example=10,
    )
    side: str = Field(
        ...,
        description="Order side: 'buy' or 'sell'.",
        example="buy",
    )
    price: Optional[float] = Field(
        None,
        gt=0,
        description="Limit price for the order; omit for market orders.",
        example=150.0,
    )

    @validator("side")
    def _validate_side(cls, v: str) -> str:
        """Restrict side to allowed values."""
        allowed = {"buy", "sell"}
        if v not in allowed:
            raise ValueError(f"side must be one of {allowed}")
        return v


# --------------------------------------------------------------------------- #
# Router setup
# --------------------------------------------------------------------------- #

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