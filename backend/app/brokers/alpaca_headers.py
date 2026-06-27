"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings


def _get_alpaca_credentials() -> tuple[str, str]:
    """Retrieve Alpaca API credentials from the application settings.

    Returns:
        A tuple containing the API key and secret key.
    """
    return settings.alpaca_api_key, settings.alpaca_secret_key


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    The headers are built using the API key and secret obtained from the
    application settings.

    Returns:
        A dictionary with the required Alpaca authentication header fields.
    """
    api_key, secret_key = _get_alpaca_credentials()
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }