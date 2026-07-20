"""Rate limiting utilities for the backend API.

This module provides a shared :class:`slowapi.Limiter` instance that is
configured with sensible defaults and helper utilities for creating
strategy‑specific limiters.  The limiter is used to protect the API from
excessive traffic while keeping latency low, which is critical for the
real‑time trading strategies running on the platform.

The implementation adds type hints, configurable defaults, and a small
factory for per‑strategy limiters.  No existing behavior is changed – the
global ``limiter`` continues to use ``get_remote_address`` as its key
function.
"""

from __future__ import annotations

import os
from typing import Final

from slowapi import Limiter
from slowapi.util import get_remote_address

# Default rate limit applied globally.  Can be overridden via the
# ``API_RATE_LIMIT`` environment variable, e.g. ``"200/minute"``.
_DEFAULT_GLOBAL_LIMIT: Final[str] = os.getenv("API_RATE_LIMIT", "100/minute")

# Global limiter used throughout the application.
limiter: Limiter = Limiter(key_func=get_remote_address, default_limits=[_DEFAULT_GLOBAL_LIMIT])


def get_strategy_limiter(strategy_name: str, limit: str | None = None) -> Limiter:
    """Create a limiter scoped to a particular trading strategy.

    Args:
        strategy_name: Identifier for the strategy (e.g. ``"mean_rev_20_1.5"``).
        limit: Optional explicit limit string (e.g. ``"50/second"``).  If
            omitted the global default is used.

    Returns:
        A :class:`slowapi.Limiter` instance with a strategy‑specific key
        function that combines the remote address and the strategy name.
    """
    if not strategy_name:
        raise ValueError("strategy_name must be a non‑empty string")

    effective_limit = limit or _DEFAULT_GLOBAL_LIMIT

    def _strategy_key_func(request) -> str:  # pragma: no cover
        """Generate a unique key per client and strategy."""
        remote = get_remote_address(request)
        return f"{remote}:{strategy_name}"

    return Limiter(key_func=_strategy_key_func, default_limits=[effective_limit])


def attach_limiter_to_app(app) -> None:
    """Attach the global ``limiter`` to a FastAPI or Starlette app.

    This helper centralises the registration logic and ensures that the
    limiter middleware is added exactly once.

    Args:
        app: The FastAPI or Starlette application instance.
    """
    if hasattr(app, "state") and getattr(app.state, "limiter_attached", False):
        return

    limiter.init_app(app)  # type: ignore[attr-defined]
    setattr(app.state, "limiter_attached", True)