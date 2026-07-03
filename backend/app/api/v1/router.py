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

# Constants
ROUTERS_TO_INCLUDE = [
    auth.router,
    accounts.router,
    orders.router,
    positions.router,
    trades.router,
    strategies.router,
    backtests.router,
    comparison.router,
    experiments.router,
    ml.router,
    risk.router,
    market_data.router,
    # Underscore-prefix alias so /market_data/* and /market-data/* both resolve
    market_data.router_underscore,
    analytics.router,
    agents.router,
    notifications.router,
    archive.router,
    improvements.router,
    monitoring.router,
    options_router,
    regime_router,
    audit_log_router,
    integrations.router,
    pipeline.router,
    leaderboard.router,
    releases.router,
    bots_router,
    scanners_router,
]

api_router = APIRouter()
for _router in ROUTERS_TO_INCLUDE:
    api_router.include_router(_router)