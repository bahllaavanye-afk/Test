"""API v1 router — mounts all sub-routers."""
from fastapi import APIRouter

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

def _validate_router(router_obj: APIRouter, name: str) -> None:
    """Validate that the provided object is an APIRouter instance.

    Args:
        router_obj: The router to validate.
        name: Descriptive name used in error messages.

    Raises:
        ValueError: If router_obj is not an APIRouter instance.
    """
    if not isinstance(router_obj, APIRouter):
        raise ValueError(f"Router '{name}' must be an instance of APIRouter, got {type(router_obj)}")

api_router = APIRouter()

_validate_router(auth.router, "auth.router")
api_router.include_router(auth.router)

_validate_router(accounts.router, "accounts.router")
api_router.include_router(accounts.router)

_validate_router(orders.router, "orders.router")
api_router.include_router(orders.router)

_validate_router(positions.router, "positions.router")
api_router.include_router(positions.router)

_validate_router(trades.router, "trades.router")
api_router.include_router(trades.router)

_validate_router(strategies.router, "strategies.router")
api_router.include_router(strategies.router)

_validate_router(backtests.router, "backtests.router")
api_router.include_router(backtests.router)

_validate_router(comparison.router, "comparison.router")
api_router.include_router(comparison.router)

_validate_router(experiments.router, "experiments.router")
api_router.include_router(experiments.router)

_validate_router(ml.router, "ml.router")
api_router.include_router(ml.router)

_validate_router(risk.router, "risk.router")
api_router.include_router(risk.router)

_validate_router(market_data.router, "market_data.router")
api_router.include_router(market_data.router)

# Underscore-prefix alias so /market_data/* and /market-data/* both resolve
_validate_router(market_data.router_underscore, "market_data.router_underscore")
api_router.include_router(market_data.router_underscore)

_validate_router(analytics.router, "analytics.router")
api_router.include_router(analytics.router)

_validate_router(agents.router, "agents.router")
api_router.include_router(agents.router)

_validate_router(notifications.router, "notifications.router")
api_router.include_router(notifications.router)

_validate_router(archive.router, "archive.router")
api_router.include_router(archive.router)

_validate_router(improvements.router, "improvements.router")
api_router.include_router(improvements.router)

_validate_router(monitoring.router, "monitoring.router")
api_router.include_router(monitoring.router)

_validate_router(options_router, "options_router")
api_router.include_router(options_router)

_validate_router(regime_router, "regime_router")
api_router.include_router(regime_router)

_validate_router(audit_log_router, "audit_log_router")
api_router.include_router(audit_log_router)

_validate_router(integrations.router, "integrations.router")
api_router.include_router(integrations.router)

_validate_router(pipeline.router, "pipeline.router")
api_router.include_router(pipeline.router)

_validate_router(leaderboard.router, "leaderboard.router")
api_router.include_router(leaderboard.router)

_validate_router(releases.router, "releases.router")
api_router.include_router(releases.router)

_validate_router(bots_router, "bots_router")
api_router.include_router(bots_router)

_validate_router(scanners_router, "scanners_router")
api_router.include_router(scanners_router)

_validate_router(discord_router, "discord_router")
api_router.include_router(discord_router)

_validate_router(webhooks_router, "webhooks_router")
api_router.include_router(webhooks_router)