"""API v1 router — mounts all sub-routers."""
import logging
import time
import json
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

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

# Configure structured logger
_logger = logging.getLogger("quantedge.api")
if not _logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":%(message)s}',
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
_logger.setLevel(logging.INFO)


api_router = APIRouter()


@api_router.middleware("http")
async def log_request(request: Request, call_next):
    """Log key metrics for each request."""
    start_time = time.time()
    response: Response = await call_next(request)
    duration = time.time() - start_time

    # Prepare base log payload
    log_payload = {
        "path": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "duration_ms": round(duration * 1000, 2),
    }

    # Attempt to extract signal count and P&L from JSON responses
    if isinstance(response, JSONResponse):
        try:
            body = json.loads(response.body)
            # Expected keys; adapt if different naming conventions are used elsewhere
            if isinstance(body, dict):
                if "signal_count" in body:
                    log_payload["signal_count"] = body["signal_count"]
                if "pnl" in body:
                    log_payload["pnl"] = body["pnl"]
        except Exception:
            # Silently ignore parsing errors – logging should not interfere with request handling
            pass

    _logger.info(json.dumps(log_payload))
    return response


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