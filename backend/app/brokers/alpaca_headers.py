"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
import logging
from app.config import settings


logger = logging.getLogger(__name__)


class AlpacaHeaderError(Exception):
    """Raised when Alpaca authentication headers cannot be constructed."""


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Raises:
        AlpacaHeaderError: If required credentials are missing or any other
            error occurs while building the header dictionary.
    """
    try:
        return {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to construct Alpaca authentication headers",
            extra={"error": str(exc)},
            exc_info=True,
        )
        raise AlpacaHeaderError("Unable to build Alpaca headers") from exc