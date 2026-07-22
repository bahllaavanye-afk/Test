"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
import time
import logging
from app.config import settings

_logger = logging.getLogger("monitoring")


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Logs structured metrics at INFO level:
        - signal_count: placeholder (None) – to be filled by caller if needed
        - execution_time: time taken to build the header dict (seconds)
        - pnl: placeholder (None) – to be filled by caller if needed
    """
    start_time = time.time()
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    elapsed = time.time() - start_time
    _logger.info(
        "alpaca_headers_generated",
        signal_count=None,
        execution_time=elapsed,
        pnl=None,
    )
    return headers