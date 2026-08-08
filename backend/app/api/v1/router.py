"""API v1 router — mounts all sub‑routers.

This module aggregates the individual feature routers for version 1 of the
public API and registers them with a top‑level :class:`fastapi.APIRouter`
instance. The resulting ``api_router`` is imported by the main FastAPI
application.
"""
import logging
from typing import List, Optional, Tuple, Any

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

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

# Log message templates
LOG_WARN_ROUTER_NONE: str = "Router %s is None and will be skipped."
LOG_WARN_ROUTER_INVALID: str = "Router definition %s is invalid and will be skipped."
LOG_WARN_NO_DEFINITIONS: str = "No router definitions provided; nothing to include."
LOG_ERROR_INCLUDE_ROUTER: str = "Failed to include router %s: %s"

# Router definitions (router instance, human‑readable name)
_ROUTER_DEFINITIONS: List[Tuple[Optional[APIRouter], str]] = [
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

logger = logging.getLogger(__name__)

api_router: APIRouter = APIRouter()


def _include(router_obj: Optional[APIRouter], name: str) -> None:
    """Safely include a sub‑router into the top‑level ``api_router``.

    Args:
        router_obj: The router to include. If ``None`` or an empty collection
            the function logs a warning and returns without raising an exception.
        name: Human‑readable identifier for the router, used in log messages.

    The function catches any unexpected exception during inclusion, logs the
    error, and continues processing the remaining routers.
    """
    if router_obj is None:
        logger.warning(LOG_WARN_ROUTER_NONE, name)
        return

    # Guard against empty iterable objects that are not APIRouter instances
    if isinstance(router_obj, (list, tuple, set, dict)) and not router_obj:
        logger.warning(LOG_WARN_ROUTER_NONE, name)
        return

    try:
        api_router.include_router(router_obj)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover
        logger.error(LOG_ERROR_INCLUDE_ROUTER, name, exc)


def _process_router_definitions(definitions: Optional[List[Tuple[Any, Any]]]) -> None:
    """Iterate over router definitions handling edge cases.

    - Skips ``None`` or empty definition lists.
    - Validates each entry is a tuple with at least two elements.
    - Delegates inclusion to ``_include``.
    """
    if not definitions:
        logger.warning(LOG_WARN_NO_DEFINITIONS)
        return

    for entry in definitions:
        if not isinstance(entry, tuple) or len(entry) < 2:
            logger.warning(LOG_WARN_ROUTER_INVALID, entry)
            continue

        router_obj, name = entry[0], entry[1]
        _include(router_obj, str(name))


# Process all router definitions safely
_process_router_definitions(_ROUTER_DEFINITIONS)