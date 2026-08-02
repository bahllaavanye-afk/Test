"""Shared slowapi rate limiter instance with strategy-specific utilities."""
from fastapi import Request, HTTPException, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from typing import Callable
import os

# Base limiter used across the application
limiter = Limiter(key_func=get_remote_address)

# Default strategy rate limit; can be overridden via environment variable
DEFAULT_STRATEGY_LIMIT = os.getenv("STRATEGY_RATE_LIMIT", "60/minute")

def strategy_limit(limit: str = DEFAULT_STRATEGY_LIMIT) -> Callable:
    """
    Decorator for FastAPI endpoints that serve strategy signals.

    Applies a configurable rate limit and enforces a confirmation header
    ``X-Strategy-Confirm`` with value ``true``. This adds an extra filter to
    tighten entry conditions for signal generation.

    Example:
        @app.get("/signal")
        @strategy_limit()
        async def get_signal(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @limiter.limit(limit)
        async def wrapper(*args, **kwargs):
            # Locate the Request object among args or kwargs
            request: Request | None = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if request is None:
                request = kwargs.get("request")
            if not isinstance(request, Request):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Request object is required for strategy limit validation.",
                )
            # Confirmation filter
            confirm = request.headers.get("X-Strategy-Confirm", "").lower()
            if confirm != "true":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Strategy confirmation header missing or invalid.",
                )
            return await func(*args, **kwargs)

        return wrapper

    return decorator

# Register the default handler for rate‑limit exceedance
limiter._exception_handler = _rate_limit_exceeded_handler