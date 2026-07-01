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
        RuntimeError: If API credentials are missing or cannot be retrieved.
    """
    try:
        api_key = settings.alpaca_api_key
        secret_key = settings.alpaca_secret_key

        if not api_key or not secret_key:
            raise ValueError("Alpaca API credentials are not set or are empty")
    except Exception as exc:
        logger.error("Failed to construct Alpaca authentication headers: %s", exc, exc_info=True)
        raise RuntimeError("Unable to build Alpaca authentication headers") from exc

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }