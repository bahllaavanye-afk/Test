"""Shared slowapi rate limiter instance with utilities for strategy endpoint limits.

This module provides a globally shared :class:`slowapi.Limiter` instance and helper
functions/decorators to apply consistent rate‑limiting policies to trading‑signal
endpoints. Tightening entry conditions and improving signal quality often relies
on preventing excessive request bursts that could overwhelm downstream services.
"""

from typing import Callable, Any
from functools import wraps

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Global limiter used across the application.
limiter = Limiter(key_func=get_remote_address)

# Default limits for strategy‑related endpoints.
# Adjust these values based on the free‑tier constraints of any external services.
DEFAULT_STRATEGY_LIMIT = "30/minute"  # 30 requests per minute per IP


def limit_strategy(
    limit: str = DEFAULT_STRATEGY_LIMIT,
    description: str = "Rate limit for strategy signal generation",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to apply a rate limit to a FastAPI/Starlette endpoint.

    Args:
        limit: A rate‑limit string understood by *slowapi* (e.g. ``"30/minute"``).
        description: Human‑readable description used in HTTP ``429`` responses.

    Returns:
        A decorator that wraps the endpoint with ``limiter.limit`` and provides a
        consistent error message when the limit is exceeded.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        @limiter.limit(limit, error_message=description)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except RateLimitExceeded as exc:
                # Re‑raise with a clearer message for downstream handling.
                raise RateLimitExceeded(
                    detail=f"Rate limit exceeded: {description}. "
                    f"Allowed: {limit}. Please retry later."
                ) from exc

        return wrapper

    return decorator


def remaining_quota(request) -> int:
    """Return the number of remaining requests for the current client.

    This helper can be used inside endpoint logic to add dynamic confirmation
    filters (e.g., only emit a signal if enough quota remains).

    Args:
        request: The Starlette/FastAPI request object.

    Returns:
        Remaining request count for the configured ``DEFAULT_STRATEGY_LIMIT``.
    """
    # ``limiter``, when called with a request, provides a ``RateLimit`` object.
    # The attribute ``remaining`` holds the count of allowed calls left.
    limit = limiter._parse_limit(DEFAULT_STRATEGY_LIMIT)  # type: ignore
    key = limiter.key_func(request)
    # The internal storage is a ``MemoryStorage`` by default.
    # ``get` returns a tuple (reset_time, remaining). Use ``remaining``.
    _, remaining = limiter.storage.get(key, limit)  # type: ignore
    return remaining