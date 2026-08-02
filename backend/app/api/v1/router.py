"""API v1 router — mounts all sub-routers."""
from fastapi import APIRouter
import logging

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

_logger = logging.getLogger(__name__)

api_router = APIRouter()


def _safe_include(router_obj):
    """Include a router if it is a valid APIRouter instance.

    Handles None inputs and logs any unexpected errors without breaking
    the router initialization.
    """
    if router_obj is None:
        _logger.debug("Attempted to include a None router; skipping.")
        return
    try:
        api_router.include_router(router_obj)
    except Exception as exc:  # pragma: no cover
        _logger.warning(f"Failed to include router {router_obj}: {exc}")


# Core routers
_safe_include(auth.router)
_safe_include(accounts.router)
_safe_include(orders.router)
_safe_include(positions.router)
_safe_include(trades.router)

_safe_include(strategies.router)

_safe_include(backtests.router)
_safe_include(comparison.router)
_safe_include(experiments.router)
_safe_include(ml.router)
_safe_include(risk.router)
_safe_include(market_data.router)
# Underscore-prefix alias so /market_data/* and /market-data/* both resolve
_safe_include(market_data.router_underscore)
_safe_include(analytics.router)
_safe_include(agents.router)
_safe_include(notifications.router)
_safe_include(archive.router)
_safe_include(improvements.router)
_safe_include(monitoring.router)
_safe_include(options_router)
_safe_include(regime_router)
_safe_include(audit_log_router)
_safe_include(integrations.router)
_safe_include(pipeline.router)
_safe_include(leaderboard.router)
_safe_include(releases.router)
_safe_include(bots_router)
_safe_include(scanners_router)
_safe_include(discord_router)
_safe_include(webhooks_router)