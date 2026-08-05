"""API v1 router — mounts all sub-routers."""
import logging
from typing import Iterable, Tuple, Optional

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

api_router = APIRouter()


def _include(router_obj: Optional[APIRouter], name: str) -> None:
    """Safely include a sub‑router, handling None, empty, or invalid inputs.

    Args:
        router_obj: The router instance to include. May be None or an invalid type.
        name: Human‑readable identifier for logging.
    """
    if router_obj is None:
        logger.warning("Router %s is None and will be skipped.", name)
        return

    # Guard against accidental passing of empty iterables or wrong types.
    if not isinstance(router_obj, APIRouter):
        logger.warning(
            "Router %s is not an APIRouter instance (type=%s); skipping inclusion.",
            name,
            type(router_obj).__name__,
        )
        return

    try:
        api_router.include_router(router_obj)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to include router %s: %s", name, exc)


# List of (router, name) tuples for systematic inclusion.
_routers: Iterable[Tuple[Optional[APIRouter], str]] = [
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

if not _routers:
    logger.warning("No sub‑routers defined; API router will have no endpoints.")
else:
    for r, n in _routers:
        _include(r, n)