"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings


def _get_alpaca_credentials() -> tuple[str, str]:
    """Retrieve Alpaca API credentials from settings.

    Returns
    -------
    tuple[str, str]
        A tuple containing the API key and secret key.
    """
    return settings.alpaca_api_key, settings.alpaca_secret_key


def _format_alpaca_header(api_key: str, secret_key: str) -> dict[str, str]:
    """Construct the authentication header dictionary for Alpaca.

    Parameters
    ----------
    api_key : str
        The Alpaca API key.
    secret_key : str
        The Alpaca secret key.

    Returns
    -------
    dict[str, str]
        A dictionary suitable for use as HTTP headers.
    """
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests."""
    api_key, secret_key = _get_alpaca_credentials()
    return _format_alpaca_header(api_key, secret_key)