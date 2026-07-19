"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings


def _get_alpaca_credentials() -> tuple[str, str]:
    """Retrieve Alpaca API credentials from the application settings."""
    return settings.alpaca_api_key, settings.alpaca_secret_key


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    The credentials are obtained via :func:`_get_alpaca_credentials` and
    formatted into the header dictionary expected by Alpaca.
    """
    api_key, secret_key = _get_alpaca_credentials()
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }