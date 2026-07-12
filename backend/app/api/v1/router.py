"""API v1 router — mounts all sub-routers."""
import logging
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

logger = logging.getLogger(__name__)

def _safe_include(parent: APIRouter, child, name: str) -> None:
    """Include a sub‑router safely.

    - Skips inclusion if the child router is ``None``.
    - Catches unexpected exceptions, logs them, and continues
      so that one faulty sub‑router does not bring down the whole API.
    """
    if child is None:
        logger.warning("Router '%s' is None and will not be included.", name)
        return
    try:
        parent.include_router(child)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to include router '%s': %s", name, exc, exc_info=True
        )

api_router = APIRouter()

_safe_include(api_router, auth.router, "auth")
_safe_include(api_router, accounts.router, "accounts")
_safe_include(api_router, orders.router, "orders")
_safe_include(api_router, positions.router, "positions")
_safe_include(api_router, trades.router, "trades")
_safe_include(api_router, strategies.router, "strategies")
_safe_include(api_router, backtests.router, "backtests")
_safe_include(api_router, comparison.router, "comparison")
_safe_include(api_router, experiments.router, "experiments")
_safe_include(api_router, ml.router, "ml")
_safe_include(api_router, risk.router, "risk")
_safe_include(api_router, market_data.router, "market_data")
# Underscore‑prefix alias so /market_data/* and /market-data/* both resolve
_safe_include(
    api_router,
    getattr(market_data, "router_underscore", None),
    "market_data.router_underscore",
)
_safe_include(api_router, analytics.router, "analytics")
_safe_include(api_router, agents.router, "agents")
_safe_include(api_router, notifications.router, "notifications")
_safe_include(api_router, archive.router, "archive")
_safe_include(api_router, improvements.router, "improvements")
_safe_include(api_router, monitoring.router, "monitoring")
_safe_include(api_router, options_router, "options")
_safe_include(api_router, regime_router, "regime")
_safe_include(api_router, audit_log_router, "audit_log")
_safe_include(api_router, integrations.router, "integrations")
_safe_include(api_router, pipeline.router, "pipeline")
_safe_include(api_router, leaderboard.router, "leaderboard")
_safe_include(api_router, releases.router, "releases")
_safe_include(api_router, bots_router, "bots")
_safe_include(api_router, scanners_router, "scanners")
_safe_include(api_router, discord_router, "discord")