"""API v1 router — mounts all sub-routers."""
from fastapi import APIRouter

from app.api.v1 import (
    accounts,
    agents,
    analytics,
    archive,
    auth,
    backtests,
    comparison,
    experiments,
    improvements,
    integrations,
    leaderboard,
    market_data,
    ml,
    monitoring,
    notifications,
    orders,
    pipeline,
    positions,
    releases,
    risk,
    strategies,
    trades,
)
from app.api.v1.agent_logs import router as agent_logs_router
from app.api.v1.alerts import router as alerts_router
from app.api.v1.audit_log import router as audit_log_router
from app.api.v1.bots import router as bots_router
from app.api.v1.options import router as options_router
from app.api.v1.regime import router as regime_router
from app.api.v1.scanners import router as scanners_router
from app.tasks.slack_handler import router as slack_router

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(accounts.router)
api_router.include_router(orders.router)
api_router.include_router(positions.router)
api_router.include_router(trades.router)
api_router.include_router(strategies.router)
api_router.include_router(backtests.router)
api_router.include_router(comparison.router)
api_router.include_router(experiments.router)
api_router.include_router(ml.router)
api_router.include_router(risk.router)
api_router.include_router(market_data.router)
# Underscore-prefix alias so /market_data/* and /market-data/* both resolve
api_router.include_router(market_data.router_underscore)
api_router.include_router(alerts_router)
api_router.include_router(analytics.router)
api_router.include_router(agents.router)
api_router.include_router(notifications.router)
api_router.include_router(archive.router)
api_router.include_router(improvements.router)
api_router.include_router(monitoring.router)
api_router.include_router(options_router)
api_router.include_router(regime_router)
api_router.include_router(audit_log_router)
api_router.include_router(agent_logs_router)
api_router.include_router(integrations.router)
api_router.include_router(pipeline.router)
api_router.include_router(leaderboard.router)
api_router.include_router(releases.router)
api_router.include_router(bots_router)
api_router.include_router(scanners_router)
api_router.include_router(slack_router)
from app.api.v1.promotions import router as promotions_router

api_router.include_router(promotions_router)
from app.api.v1.copy_trading import router as copy_trading_router
from app.api.v1.tasks import router as tasks_router

api_router.include_router(tasks_router)
api_router.include_router(copy_trading_router)
from app.api.v1.polymarket import router as polymarket_router
from app.api.v1.kalshi import router as kalshi_router
api_router.include_router(polymarket_router)
api_router.include_router(kalshi_router)
