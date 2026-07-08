"""API v1 router — mounts all sub-routers with lazy imports for faster startup."""
from fastapi import APIRouter
from importlib import import_module

# Mapping of module path to the attribute that holds the FastAPI router.
# Most modules expose a `router` attribute; a few expose differently named attributes.
_ROUTER_SPECS = [
    ("app.api.v1.auth", "router"),
    ("app.api.v1.accounts", "router"),
    ("app.api.v1.orders", "router"),
    ("app.api.v1.positions", "router"),
    ("app.api.v1.trades", "router"),
    ("app.api.v1.strategies", "router"),
    ("app.api.v1.backtests", "router"),
    ("app.api.v1.comparison", "router"),
    ("app.api.v1.experiments", "router"),
    ("app.api.v1.ml", "router"),
    ("app.api.v1.risk", "router"),
    ("app.api.v1.market_data", "router"),
    # Underscore-prefixed alias so /market_data/* and /market-data/* both resolve
    ("app.api.v1.market_data", "router_underscore"),
    ("app.api.v1.analytics", "router"),
    ("app.api.v1.agents", "router"),
    ("app.api.v1.notifications", "router"),
    ("app.api.v1.archive", "router"),
    ("app.api.v1.improvements", "router"),
    ("app.api.v1.monitoring", "router"),
    ("app.api.v1.options", "router"),
    ("app.api.v1.regime", "router"),
    ("app.api.v1.audit_log", "router"),
    ("app.api.v1.integrations", "router"),
    ("app.api.v1.pipeline", "router"),
    ("app.api.v1.leaderboard", "router"),
    ("app.api.v1.releases", "router"),
    ("app.api.v1.bots", "router"),
    ("app.api.v1.scanners", "router"),
    ("app.api.v1.discord_interactions", "router"),
]

api_router = APIRouter()

def _include_router(module_path: str, attr_name: str) -> None:
    """Import the module and include its router on the main APIRouter."""
    module = import_module(module_path)
    router_obj = getattr(module, attr_name)
    api_router.include_router(router_obj)

# Dynamically include all sub‑routers. This keeps import time low and centralises
# the inclusion logic, making future additions easier and reducing boilerplate.
for _module_path, _attr_name in _ROUTER_SPECS:
    _include_router(_module_path, _attr_name)