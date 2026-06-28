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
    application settings. Validates that the credentials are non‑empty strings.

    Returns:
        A dictionary with the required Alpaca authentication header fields.

    Raises:
        ValueError: If the API key or secret key is missing or not a string.
    """
    api_key, secret_key = _get_alpaca_credentials()

    if not isinstance(api_key, str) or not api_key:
        raise ValueError("Alpaca API key must be a non-empty string.")
    if not isinstance(secret_key, str) or not secret_key:
        raise ValueError("Alpaca secret key must be a non-empty string.")

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }