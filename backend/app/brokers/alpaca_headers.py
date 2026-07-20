"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    This function validates that the required Alpaca API credentials are
    present and correctly typed before constructing the header dictionary.
    It raises a RuntimeError if the credentials are missing or malformed,
    preventing downstream API calls with invalid authentication data.

    Returns
    -------
    dict[str, str]
        The authentication headers required by Alpaca's REST API.
    """
    api_key = getattr(settings, "alpaca_api_key", None)
    secret_key = getattr(settings, "alpaca_secret_key", None)

    if not api_key or not secret_key:
        raise RuntimeError("Alpaca API credentials are not configured.")
    if not isinstance(api_key, str) or not isinstance(secret_key, str):
        raise RuntimeError("Alpaca API credentials must be strings.")

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }