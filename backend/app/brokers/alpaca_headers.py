"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings


def _build_alpaca_header_dict() -> dict[str, str]:
    """Construct the raw header dictionary using configured credentials."""
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Delegates to a private helper to keep the public interface concise and
    facilitate future extensions or testing.
    """
    return _build_alpaca_header_dict()