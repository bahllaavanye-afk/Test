"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
import logging
from typing import Dict

from app.config import settings

logger = logging.getLogger(__name__)


class AlpacaConfigurationError(RuntimeError):
    """Raised when required Alpaca configuration values are missing or invalid."""
    pass


def alpaca_headers() -> Dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Raises:
        AlpacaConfigurationError: If the API key or secret is missing or empty.
    """
    try:
        api_key = settings.alpaca_api_key
        secret_key = settings.alpaca_secret_key
    except AttributeError as exc:
        logger.error(
            "Alpaca configuration attributes are missing.",
            extra={"exception": type(exc).__name__},
            exc_info=True,
        )
        raise AlpacaConfigurationError("Alpaca API credentials are not configured.") from exc

    if not api_key or not secret_key:
        logger.error(
            "Alpaca API credentials are empty.",
            extra={"api_key_present": bool(api_key), "secret_key_present": bool(secret_key)},
        )
        raise AlpacaConfigurationError("Alpaca API credentials must not be empty.")

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }