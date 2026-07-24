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

api_router = APIRouter()


def _safe_include(module, attr_name: str = "router") -> None:
    """
    Safely include a sub‑router into the main router.

    This helper guards against:
    * ``module`` being ``None``.
    * The requested attribute missing or being ``None``.
    * Empty collections that could raise errors in FastAPI (unlikely but defensive).

    Parameters
    ----------
    module: Any
        The imported module that should expose a FastAPI router.
    attr_name: str, optional
        Name of the attribute containing the router (default ``"router"``).

    Returns
    -------
    None
    """
    if module is None:
        return
    router_obj = getattr(module, attr_name, None)
    if router_obj is None:
        return
    # FastAPI's include_router tolerates empty routers, but we defensively skip None.
    api_router.include_router(router_obj)


# Core sub‑routers
_safe_include(auth)
_safe_include(accounts)
_safe_include(orders)
_safe_include(positions)
_safe_include(trades)

# Strategy‑related routers
_safe_include(strategies)

# Additional feature routers
_safe_include(backtests)
_safe_include(comparison)
_safe_include(experiments)
_safe_include(ml)
_safe_include(risk)
_safe_include(market_data)
# Underscore‑prefix alias so /market_data/* and /market-data/* both resolve
_safe_include(market_data, "router_underscore")
_safe_include(analytics)
_safe_include(agents)
_safe_include(notifications)
_safe_include(archive)
_safe_include(improvements)
_safe_include(monitoring)
_safe_include(options_router, "router")
_safe_include(regime_router, "router")
_safe_include(audit_log_router, "router")
_safe_include(integrations)
_safe_include(pipeline)
_safe_include(leaderboard)
_safe_include(releases)
_safe_include(bots_router, "router")
_safe_include(scanners_router, "router")
_safe_include(discord_router, "router")
_safe_include(webhooks_router, "router")