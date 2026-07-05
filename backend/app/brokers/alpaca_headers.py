"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.

Adds structured logging (INFO level) for key metrics:
- signal_count: number of signals processed in the current cycle
- execution_time_ms: time taken to build the header dict
- pnl: profit & loss associated with the current cycle (if applicable)
"""

import logging
import time
from app.config import settings

_logger = logging.getLogger(__name__)


def alpaca_headers(
    signal_count: int = 0,
    execution_time_ms: float | None = None,
    pnl: float = 0.0,
) -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Parameters
    ----------
    signal_count : int, optional
        Count of signals processed in the current trading cycle. Defaults to 0.
    execution_time_ms : float | None, optional
        Execution time in milliseconds for building the header. If None,
        the function measures the time internally. Defaults to None.
    pnl : float, optional
        Profit & loss associated with the current cycle. Defaults to 0.0.

    Returns
    -------
    dict[str, str]
        Authentication header dictionary for Alpaca API calls.
    """
    start = time.time()
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    # Determine execution time if not supplied
    if execution_time_ms is None:
        execution_time_ms = (time.time() - start) * 1000.0

    # Structured logging of key metrics
    _logger.info(
        "Alpaca headers generated",
        extra={
            "signal_count": signal_count,
            "execution_time_ms": execution_time_ms,
            "pnl": pnl,
        },
    )
    return headers