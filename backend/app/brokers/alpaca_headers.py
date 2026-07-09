"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
import logging
from app.config import settings

logger = logging.getLogger(__name__)


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Raises:
        AttributeError: If required configuration attributes are missing.
        ValueError: If API key or secret is empty or not provided.
    """
    try:
        api_key = settings.alpaca_api_key
        secret_key = settings.alpaca_secret_key
    except AttributeError as exc:
        logger.error(
            "Alpaca configuration attributes are missing",
            exc_info=True,
            extra={"exception": str(exc)},
        )
        raise

    if not api_key or not secret_key:
        logger.error(
            "Alpaca API credentials are not set or are empty",
            extra={"api_key_present": bool(api_key), "secret_key_present": bool(secret_key)},
        )
        raise ValueError("Alpaca API key and secret must be configured")

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }