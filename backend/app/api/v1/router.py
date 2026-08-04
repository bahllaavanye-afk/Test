"""API v1 router — mounts all sub-routers."""
import logging
from typing import Iterable, Tuple, Any

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


def _include(router_obj: Any, name: str) -> None:
    """Safely include a sub‑router, handling None or invalid inputs."""
    if not name:
        logger.warning("Router name is empty; skipping inclusion.")
        return
    if router_obj is None:
        logger.warning("Router %s is None and will be skipped.", name)
        return
    # Basic type guard: FastAPI routers should be instances of APIRouter
    if not isinstance(router_obj, APIRouter):
        logger.warning(
            "Router %s is not an APIRouter instance (%s); skipping.", name, type(router_obj)
        )
        return
    try:
        api_router.include_router(router_obj)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to include router %s: %s", name, exc)


# List of (router, name) tuples for systematic inclusion
_routers: Iterable[Tuple[Any, str]] = [
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
    logger.warning("No routers defined for inclusion; API will have no endpoints.")
else:
    for entry in _routers:
        # Guard against malformed entries (e.g., not a tuple of length 2)
        if not isinstance(entry, tuple) or len(entry) != 2:
            logger.warning("Malformed router entry %s; expected (router, name) tuple.", entry)
            continue
        router_obj, name = entry
        _include(router_obj, name)