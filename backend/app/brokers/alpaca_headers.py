"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from functools import lru_cache
from app.config import settings

# Constants
ALPACA_API_KEY_HEADER = "APCA-API-KEY-ID"
ALPACA_API_SECRET_HEADER = "APCA-API-SECRET-KEY"
CACHE_MAXSIZE = 1


@lru_cache(maxsize=CACHE_MAXSIZE)
def _cached_alpaca_headers() -> dict[str, str]:
    """Build and cache the Alpaca authentication header dict."""
    return {
        ALPACA_API_KEY_HEADER: settings.alpaca_api_key,
        ALPACA_API_SECRET_HEADER: settings.alpaca_secret_key,
    }


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    The underlying header dict is cached for performance; a shallow copy is
    returned to prevent accidental mutation of the cached object.
    """
    return _cached_alpaca_headers().copy()