"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
import logging
from app.config import settings


logger = logging.getLogger(__name__)


class AlpacaAuthError(Exception):
    """Raised when Alpaca authentication headers cannot be built due to missing or invalid credentials."""
    pass


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Raises:
        AlpacaAuthError: If required authentication settings are missing or empty.
    """
    try:
        api_key = settings.alpaca_api_key
        secret_key = settings.alpaca_secret_key
    except AttributeError as exc:
        logger.error(
            "Alpaca configuration missing required attributes",
            exc_info=True,
            extra={"missing_attribute": str(exc)},
        )
        raise AlpacaAuthError("Alpaca API key or secret is not configured") from exc

    if not api_key or not secret_key:
        logger.error(
            "Alpaca credentials are empty or None",
            extra={"api_key_present": bool(api_key), "secret_key_present": bool(secret_key)},
        )
        raise AlpacaAuthError("Alpaca API key or secret is empty")

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }