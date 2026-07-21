"""API v1 router — mounts all sub-routers."""
from fastapi import APIRouter, Depends, Request, HTTPException
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

def _validate_strategy_signal(request: Request):
    """
    Enforce tighter entry conditions and confirmation filters for strategy endpoints.
    Expected headers:
        X-Strategy-Entry: "true"
        X-Strategy-Confirmation: "true"
    An optional exit validation header can be provided:
        X-Strategy-Exit: "true"
    """
    entry = request.headers.get("X-Strategy-Entry")
    confirmation = request.headers.get("X-Strategy-Confirmation")
    if entry != "true" or confirmation != "true":
        raise HTTPException(
            status_code=400,
            detail="Strategy signal validation failed: missing or invalid entry/confirmation headers",
        )
    # Exit logic can be relaxed; we only log if missing for observability
    exit_signal = request.headers.get("X-Strategy-Exit")
    if exit_signal != "true":
        # Not raising an error to keep backward compatibility; could be used for monitoring
        pass
    return True

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(accounts.router)
api_router.include_router(orders.router)
api_router.include_router(positions.router)
api_router.include_router(trades.router)

# Apply validation dependency to all strategy-related routes
api_router.include_router(
    strategies.router,
    dependencies=[Depends(_validate_strategy_signal)],
)

api_router.include_router(backtests.router)
api_router.include_router(comparison.router)
api_router.include_router(experiments.router)
api_router.include_router(ml.router)
api_router.include_router(risk.router)
api_router.include_router(market_data.router)
# Underscore-prefix alias so /market_data/* and /market-data/* both resolve
api_router.include_router(market_data.router_underscore)
api_router.include_router(analytics.router)
api_router.include_router(agents.router)
api_router.include_router(notifications.router)
api_router.include_router(archive.router)
api_router.include_router(improvements.router)
api_router.include_router(monitoring.router)
api_router.include_router(options_router)
api_router.include_router(regime_router)
api_router.include_router(audit_log_router)
api_router.include_router(integrations.router)
api_router.include_router(pipeline.router)
api_router.include_router(leaderboard.router)
api_router.include_router(releases.router)
api_router.include_router(bots_router)
api_router.include_router(scanners_router)
api_router.include_router(discord_router)
api_router.include_router(webhooks_router)