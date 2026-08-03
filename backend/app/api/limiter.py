"""Shared slowapi rate limiter instance with enhanced configuration."""
import os
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

def _client_key_func(request):
    """
    Resolve a unique client identifier for rate limiting.

    Preference order:
    1. ``X-User-ID`` header (if present) – useful for authenticated internal services.
    2. Remote IP address – fallback for external callers.
    """
    user_id = request.headers.get("X-User-ID")
    if user_id:
        return f"user:{user_id}"
    return get_remote_address(request)

# Default limit can be overridden via environment variable, e.g., "500 per hour"
DEFAULT_LIMITS = os.getenv("RATE_LIMIT_DEFAULT", "200 per minute")

try:
    limiter = Limiter(key_func=_client_key_func, default_limits=[DEFAULT_LIMITS])
except Exception as exc:  # pragma: no cover
    # Guard against mis‑configuration that could crash the app at import time.
    logger.error("Failed to initialise rate limiter: %s", exc)
    # Fallback to a permissive limiter to avoid blocking the entire service.
    limiter = Limiter(key_func=_client_key_func)