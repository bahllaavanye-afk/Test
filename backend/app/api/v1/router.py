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

def _safe_include(parent: APIRouter, child) -> None:
    """
    Safely include a sub‑router into the parent router.

    Handles:
    * ``None`` values – ignored.
    * Objects that are themselves ``APIRouter`` instances.
    * Objects exposing a ``router`` attribute that is an ``APIRouter``.
    * Duplicate inclusions – ignored to avoid off‑by‑one registration errors.
    """
    if child is None:
        return

    # Resolve the actual APIRouter instance.
    subrouter = child if isinstance(child, APIRouter) else getattr(child, "router", None)

    if not isinstance(subrouter, APIRouter):
        return

    # Prevent duplicate registration (off‑by‑one condition).
    if getattr(subrouter, "_included_in_parent", None) is parent:
        return

    parent.include_router(subrouter)
    # Mark the subrouter as already included in this parent.
    setattr(subrouter, "_included_in_parent", parent)


api_router = APIRouter()

# Core sub‑routers
_safe_include(api_router, auth.router)
_safe_include(api_router, accounts.router)
_safe_include(api_router, orders.router)
_safe_include(api_router, positions.router)
_safe_include(api_router, trades.router)
_safe_include(api_router, strategies.router)
_safe_include(api_router, backtests.router)
_safe_include(api_router, comparison.router)
_safe_include(api_router, experiments.router)
_safe_include(api_router, ml.router)
_safe_include(api_router, risk.router)
_safe_include(api_router, market_data.router)

# Underscore‑prefix alias so /market_data/* and /market-data/* both resolve
_safe_include(api_router, getattr(market_data, "router_underscore", None))

_safe_include(api_router, analytics.router)
_safe_include(api_router, agents.router)
_safe_include(api_router, notifications.router)
_safe_include(api_router, archive.router)
_safe_include(api_router, improvements.router)
_safe_include(api_router, monitoring.router)
_safe_include(api_router, options_router)
_safe_include(api_router, regime_router)
_safe_include(api_router, audit_log_router)
_safe_include(api_router, integrations.router)
_safe_include(api_router, pipeline.router)
_safe_include(api_router, leaderboard.router)
_safe_include(api_router, releases.router)
_safe_include(api_router, bots_router)
_safe_include(api_router, scanners_router)