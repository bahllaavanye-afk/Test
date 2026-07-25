"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from functools import lru_cache
from app.config import settings


@lru_cache(maxsize=1)
def _cached_alpaca_headers() -> dict[str, str]:
    """Build the Alpaca authentication headers once and cache them."""
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    The underlying header dict is cached after the first call to avoid
    repeated lookups on the settings object. A shallow copy is returned
    to prevent accidental mutation of the cached data.
    """
    return dict(_cached_alpaca_headers())