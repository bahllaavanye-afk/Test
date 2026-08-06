"""API v1 router — mounts all sub‑routers.

This module aggregates the individual feature routers for version 1 of the
public API and registers them with a top‑level :class:`fastapi.APIRouter`
instance. The resulting ``api_router`` is imported by the main FastAPI
application.
"""
import logging
from typing import List, Optional, Tuple

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

logger = logging.getLogger(__name__)

api_router: APIRouter = APIRouter()


def _include(router_obj: Optional[APIRouter], name: str) -> None:
    """Safely include a sub‑router into the top‑level ``api_router``.

    Args:
        router_obj: The router to include. If ``None`` the function logs a
            warning and returns without raising an exception.
        name: Human‑readable identifier for the router, used in log messages.

    The function catches any unexpected exception during inclusion, logs the
    error, and continues processing the remaining routers.
    """
    if router_obj is None:
        logger.warning("Router %s is None and will be skipped.", name)
        return
    try:
        api_router.include_router(router_obj)
        logger.debug("Successfully included router %s.", name)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to include router %s: %s", name, exc)


def _register_routers() -> None:
    """Register all sub‑routers with ``api_router``.

    The routers are defined as a list of ``(router, name)`` tuples. Each router
    is added via the ``_include`` helper, which isolates inclusion errors and
    logs informative messages.
    """
    routers: List[Tuple[Optional[APIRouter], str]] = [
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

    for router_obj, name in routers:
        _include(router_obj, name)


# Execute router registration at import time
_register_routers()