"""API v1 router — mounts all sub-routers."""

from fastapi import APIRouter
import importlib
from functools import lru_cache

# Lazy loader for router objects; caches imported modules to avoid repeated imports.
@lru_cache(maxsize=None)
def _load_router(module_path: str, attr: str = "router"):
    """Import a module and return the specified router attribute.

    Args:
        module_path: Full dotted path to the module containing the router.
        attr: Attribute name of the router within the module (defaults to ``router``).

    Returns:
        The FastAPI router instance defined in the target module.
    """
    module = importlib.import_module(module_path)
    return getattr(module, attr)


api_router = APIRouter()

# Mapping of module paths to router attribute names (if non‑standard).
_router_specs = [
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
    # Underscore‑prefix alias so /market_data/* and /market-data/* both resolve
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
]

# Include each sub‑router using the lazy loader.
for module_path, attr_name in _router_specs:
    api_router.include_router(_load_router(module_path, attr_name))