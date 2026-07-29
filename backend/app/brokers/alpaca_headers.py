"""Shared Alpaca authentication header builder.

All strategies and API handlers that call the Alpaca REST API should import
this function instead of duplicating the header dict inline. The builder now
includes basic validation, optional extra headers, and debug logging.
"""
import logging
from typing import Mapping, Optional

from app.config import settings

logger = logging.getLogger(__name__)

__all__ = ["alpaca_headers"]


def _validate_credentials() -> None:
    """Ensure Alpaca credentials are present and non‑empty."""
    if not getattr(settings, "alpaca_api_key", None):
        raise ValueError("Alpaca API key is not configured.")
    if not getattr(settings, "alpaca_secret_key", None):
        raise ValueError("Alpaca secret key is not configured.")


def alpaca_headers(extra_headers: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Args:
        extra_headers: Optional mapping of additional headers to merge into the
            authentication payload (e.g., custom User‑Agent).

    Returns:
        A dictionary containing the required Alpaca authentication headers plus any
        caller‑provided extras.
    """
    _validate_credentials()

    base_headers: dict[str, str] = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }

    if extra_headers:
        # Merge extra headers, allowing caller to override defaults if needed.
        base_headers.update(extra_headers)

    logger.debug("Generated Alpaca headers: %s", {k: "***" for k in base_headers})
    return base_headers