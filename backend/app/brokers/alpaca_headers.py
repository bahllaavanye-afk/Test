"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings


def _get_alpaca_credentials() -> tuple[str, str]:
    """Retrieve Alpaca API key and secret from settings.

    Returns:
        A tuple containing (api_key, secret_key).
    """
    return settings.alpaca_api_key, settings.alpaca_secret_key


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    The function builds the header dictionary using credentials obtained from
    the application settings.

    Returns:
        A dictionary with the required Alpaca authentication headers.
    """
    api_key, secret_key = _get_alpaca_credentials()
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }