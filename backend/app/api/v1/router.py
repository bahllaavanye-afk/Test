"""API v1 router — mounts all sub-routers."""
from fastapi import APIRouter
from typing import Iterable, Optional

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
)
from app.api.v1.scanners import router as scanners_router
from app.api.v1.options import router as options_router
from app.api.v1.regime import router as regime_router
from app.api.v1.audit_log import router as audit_log_router
from app.api.v1.bots import router as bots_router
from app.api.v1.discord_interactions import router as discord_router

api_router = APIRouter()

def _include_router_safely(router: Optional[APIRouter]) -> None:
    """
    Include a sub‑router only if it is a valid APIRouter instance.
    This guards against None imports or mis‑configured routers that could
    raise exceptions during application start‑up.
    """
    if isinstance(router, APIRouter):
        api_router.include_router(router)

# Primary routers
_primary_routers: Iterable[Optional[APIRouter]] = [
    getattr(auth, "router", None),
    getattr(accounts, "router", None),
    getattr(orders, "router", None),
    getattr(positions, "router", None),
    getattr(trades, "router", None),
    getattr(strategies, "router", None),
    getattr(backtests, "router", None),
    getattr(comparison, "router", None),
    getattr(experiments, "router", None),
    getattr(ml, "router", None),
    getattr(risk, "router", None),
    getattr(market_data, "router", None),
    getattr(market_data, "router_underscore", None),  # underscore alias
    getattr(analytics, "router", None),
    getattr(agents, "router", None),
    getattr(notifications, "router", None),
    getattr(archive, "router", None),
    getattr(improvements, "router", None),
    getattr(monitoring, "router", None),
    getattr(options_router, "router", options_router),  # already a router
    getattr(regime_router, "router", regime_router),
    getattr(audit_log_router, "router", audit_log_router),
    getattr(integrations, "router", None),
    getattr(pipeline, "router", None),
    getattr(leaderboard, "router", None),
    getattr(releases, "router", None),
    getattr(bots_router, "router", bots_router),
    getattr(scanners_router, "router", scanners_router),
    getattr(discord_router, "router", discord_router),
]

for r in _primary_routers:
    _include_router_safely(r)