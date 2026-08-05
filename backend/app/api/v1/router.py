"""API v1 router — mounts all sub‑routers."""
import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field, root_validator

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

logger = logging.getLogger(__name__)

api_router = APIRouter()


class RouterInclusionResult(BaseModel):
    """Result of attempting to include a sub‑router."""

    name: str = Field(
        ...,
        description="Identifier of the sub‑router being included.",
        example="auth",
    )
    included: bool = Field(
        ...,
        description="Flag indicating whether the router was successfully included.",
        example=True,
    )
    error: Optional[str] = Field(
        None,
        description="Error message if inclusion failed; omitted when inclusion succeeds.",
        example="AttributeError: 'router' not found",
    )

    @root_validator
    def check_error_consistency(cls, values):
        """Validate that error is present when inclusion fails and absent when successful."""
        included, error = values.get("included"), values.get("error")
        if included and error:
            raise ValueError("Error should be None when inclusion succeeds.")
        if not included and not error:
            raise ValueError("Error message required when inclusion fails.")
        return values


def _include(router_obj, name: str) -> RouterInclusionResult:
    """Safely include a sub‑router, handling None or invalid inputs.

    Returns a RouterInclusionResult describing the outcome.
    """
    if router_obj is None:
        logger.warning("Router %s is None and will be skipped.", name)
        return RouterInclusionResult(name=name, included=False, error="Router is None")
    try:
        api_router.include_router(router_obj)
        return RouterInclusionResult(name=name, included=True)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to include router %s: %s", name, exc)
        return RouterInclusionResult(name=name, included=False, error=str(exc))


# List of (router, name) tuples for systematic inclusion
_routers = [
    (auth.router, "auth"),
    (accounts.router, "accounts"),
    (orders.router, "orders"),
    (positions.router, "positions"),
    (trades.router, "trades"),
    (strategies.router, "strategies"),
    (backtests.router, "backtests"),
    (comparison.router, "comparison"),
    (experiments.router, "experiments"),
    (ml.router, "ml"),
    (risk.router, "risk"),
    (market_data.router, "market_data"),
    (getattr(market_data, "router_underscore", None), "market_data_underscore"),
    (analytics.router, "analytics"),
    (agents.router, "agents"),
    (notifications.router, "notifications"),
    (archive.router, "archive"),
    (improvements.router, "improvements"),
    (monitoring.router, "monitoring"),
    (options_router, "options"),
    (regime_router, "regime"),
    (audit_log_router, "audit_log"),
    (integrations.router, "integrations"),
    (pipeline.router, "pipeline"),
    (leaderboard.router, "leaderboard"),
    (releases.router, "releases"),
    (bots_router, "bots"),
    (scanners_router, "scanners"),
    (discord_router, "discord"),
    (webhooks_router, "webhooks"),
]

for r, n in _routers:
    _include(r, n)