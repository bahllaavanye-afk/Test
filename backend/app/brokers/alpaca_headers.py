"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings


def _make_auth_headers() -> dict[str, str]:
    """Construct the Alpaca authentication header mapping."""
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests."""
    return _make_auth_headers()