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
from app.api.v1.discord_interactions import router as discord_router
from app.api.v1.webhooks import router as webhooks_router

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

# ---------------------------------------------------------------------------
# Unit tests for router edge cases
# ---------------------------------------------------------------------------
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

def _create_app() -> FastAPI:
    """Create a FastAPI app with the version‑1 router mounted."""
    app = FastAPI()
    app.include_router(api_router)
    return app

def test_market_data_alias_routes():
    """Ensure both underscore and hyphen prefixes are registered."""
    app = _create_app()
    paths = {route.path for route in app.routes}
    assert any(p.startswith("/market_data") for p in paths), "Missing /market_data prefix"
    assert any(p.startswith("/market-data") for p in paths), "Missing /market-data prefix"

def test_duplicate_router_inclusion_raises():
    """Including the same sub‑router twice should raise a ValueError."""
    dummy = APIRouter()

    @dummy.get("/dup")
    def dup():
        return {"ok": True}

    app = FastAPI()
    app.include_router(dummy)
    with pytest.raises(ValueError):
        app.include_router(dummy)

def test_api_router_has_routes():
    """The main API router must expose at least one route."""
    assert len(api_router.routes) > 0, "api_router has no routes registered"