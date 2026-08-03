"""API v1 router — mounts all sub‑routers."""
import logging
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field, validator

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


class RouterInfo(BaseModel):
    """Schema describing a sub‑router to be mounted on the API."""
    name: str = Field(
        ...,
        description="Human‑readable identifier for the sub‑router.",
        example="auth",
    )
    router: APIRouter = Field(
        ...,
        description="FastAPI APIRouter instance that contains the endpoints for the sub‑router.",
    )

    @validator("router")
    def validate_router_instance(cls, v):
        """Ensure the provided router is a FastAPI APIRouter instance."""
        if not isinstance(v, APIRouter):
            raise ValueError("router must be an instance of fastapi.APIRouter")
        return v


def _include(router_obj: Optional[APIRouter], name: str) -> None:
    """Safely include a sub‑router, handling None or invalid inputs.

    Args:
        router_obj: The APIRouter instance to include. If ``None`` the router is skipped.
        name: Identifier used for logging.
    """
    if router_obj is None:
        logger.warning("Router %s is None and will be skipped.", name)
        return
    try:
        api_router.include_router(router_obj)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to include router %s: %s", name, exc)


# List of RouterInfo objects for systematic inclusion
_routers: List[RouterInfo] = [
    RouterInfo(name="auth", router=auth.router),
    RouterInfo(name="accounts", router=accounts.router),
    RouterInfo(name="orders", router=orders.router),
    RouterInfo(name="positions", router=positions.router),
    RouterInfo(name="trades", router=trades.router),
    RouterInfo(name="strategies", router=strategies.router),
    RouterInfo(name="backtests", router=backtests.router),
    RouterInfo(name="comparison", router=comparison.router),
    RouterInfo(name="experiments", router=experiments.router),
    RouterInfo(name="ml", router=ml.router),
    RouterInfo(name="risk", router=risk.router),
    RouterInfo(name="market_data", router=market_data.router),
    RouterInfo(
        name="market_data_underscore",
        router=getattr(market_data, "router_underscore", None),
    ),
    RouterInfo(name="analytics", router=analytics.router),
    RouterInfo(name="agents", router=agents.router),
    RouterInfo(name="notifications", router=notifications.router),
    RouterInfo(name="archive", router=archive.router),
    RouterInfo(name="improvements", router=improvements.router),
    RouterInfo(name="monitoring", router=monitoring.router),
    RouterInfo(name="options", router=options_router),
    RouterInfo(name="regime", router=regime_router),
    RouterInfo(name="audit_log", router=audit_log_router),
    RouterInfo(name="integrations", router=integrations.router),
    RouterInfo(name="pipeline", router=pipeline.router),
    RouterInfo(name="leaderboard", router=leaderboard.router),
    RouterInfo(name="releases", router=releases.router),
    RouterInfo(name="bots", router=bots_router),
    RouterInfo(name="scanners", router=scanners_router),
    RouterInfo(name="discord", router=discord_router),
    RouterInfo(name="webhooks", router=webhooks_router),
]

for router_info in _routers:
    _include(router_info.router, router_info.name)