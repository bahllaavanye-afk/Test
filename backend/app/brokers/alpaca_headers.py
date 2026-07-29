"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings

# Cache immutable values at import time to avoid repeated attribute lookups.
_API_KEY = settings.alpaca_api_key
_API_SECRET = settings.alpaca_secret_key


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests."""
    return {
        "APCA-API-KEY-ID": _API_KEY,
        "APCA-API-SECRET-KEY": _API_SECRET,
    }