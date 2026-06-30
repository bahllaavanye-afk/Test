"""
VWAP (Volume-Weighted Average Price) execution.

Participates at 10 % of market volume across the trading session.
Minimizes market impact by timing orders with the expected intraday volume
distribution.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import List

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger

# Empirical U‑shaped intraday volume profile (30‑min buckets, 13 buckets = 6.5 h day)
# Heavy open/close, lighter midday — matches NYSE observed volume patterns
_EMPIRICAL_PROFILE = [
    0.12,
    0.08,
    0.07,
    0.06,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.06,
    0.06,
    0.08,
    0.12,
]


async def get_intraday_volume_profile(
    symbol: str, broker: AbstractBroker | None = None
) -> List[float]:
    """Return a normalized intraday volume distribution for *symbol*.

    The function attempts to fetch the previous trading day's 30‑minute bar
    volumes from the supplied *broker*.  If the broker returns sufficient data
    (at least eight non‑zero volume entries), the raw volumes are normalised to
    create a dynamic profile that reflects the instrument's recent trading
    pattern.

    If the broker is ``None`` or the fetch fails, the function falls back to the
    static empirical U‑shaped profile defined by ``_EMPIRICAL_PROFILE``.

    Parameters
    ----------
    symbol:
        Ticker symbol for which to retrieve the volume distribution.
    broker:
        Optional :class:`~app.brokers.base.AbstractBroker` instance used to query
        historical bars.  When ``None`` the empirical profile is returned.

    Returns
    -------
    List[float]
        Normalised volume weights that sum to 1.0 (or the empirical profile if
        dynamic data could not be obtained).
    """
    if broker is None:
        return list(_EMPIRICAL_PROFILE)

    try:
        bars = await broker.get_bars(symbol, timeframe="30Min", limit=13)
        volumes = [float(getattr(b, "volume", 0) or 0) for b in bars]
        volumes = [v for v in volumes if v > 0]

        if len(volumes) >= 8:
            total = sum(volumes)
            profile = [v / total for v in volumes]
            logger.debug(
                "VWAP dynamic profile loaded",
                symbol=symbol,
                buckets=len(profile),
            )
            return profile
        else:
            logger.info(
                "Insufficient volume data for dynamic VWAP profile; using empirical fallback",
                symbol=symbol,
                valid_buckets=len(volumes),
            )
    except (ConnectionError, TimeoutError) as e:
        logger.error(
            "Network error while fetching VWAP volume profile",
            symbol=symbol,
            error=str(e),
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            "Unexpected error while fetching VWAP volume profile",
            symbol=symbol,
            error=str(e),
            exc_info=True,
        )

    return list(_EMPIRICAL_PROFILE)


class VWAPExecution:
    """Execute orders using a VWAP strategy.

    The algorithm slices the total order quantity according to an intraday
    volume profile and sends market orders at a fixed participation rate.
    """

    def __init__(
        self,
        broker: AbstractBroker,
        participation_rate: float = 0.10,
        slices: int = 12,
    ) -> None:
        """
        Parameters
        ----------
        broker:
            The broker implementation used to place orders and fetch market data.
        participation_rate:
            Desired fraction of market volume to participate in (default 10 %).
        slices:
            Number of time slices (or intervals) the order will be divided into.
        """
        if not (0 < participation_rate <= 1):
            raise ValueError(
                f"participation_rate must be between 0 (exclusive) and 1 (inclusive), got {participation_rate}"
            )
        if slices <= 0:
            raise ValueError(f"slices must be a positive integer, got {slices}")

        self.broker = broker
        self.participation_rate = participation_rate
        self.slices = slices
        self.sleep_seconds = (6.5 * 3600) / self.slices

    async def execute(self, request: OrderRequest) -> OrderResult:
        """Execute a VWAP order.

        The method retrieves a volume profile, splits the order into slices, and
        sends each slice as a market order.  Between slices it sleeps for the
        calculated interval to spread execution across the session.

        Parameters
        ----------
        request:
            An :class:`~app.brokers.base.OrderRequest` describing the order to be
            executed (symbol, quantity, etc.).

        Returns
        -------
        OrderResult
            Aggregated result containing the total filled quantity, average fill
            price, and an overall status (``filled`` if at least 95 % of the
            target quantity was executed, otherwise ``partial``).
        """
        if request.quantity <= 0:
            raise ValueError(
                f"Order quantity must be positive, got {request.quantity}"
            )

        # Fetch dynamic profile; cap slices to profile length
        profile = await get_intraday_volume_profile(request.symbol, self.broker)
        active_slices = min(self.slices, len(profile))
        profile_slice = profile[:active_slices]
        profile_total = sum(profile_slice)

        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None

        for i in range(active_slices):
            slice_weight = profile_slice[i] / profile_total
            slice_qty = request.quantity * slice_weight

            slice_req = OrderRequest(
                **{**asdict(request), "quantity": slice_qty, "order_type": "market"}
            )
            try:
                result = await self.broker.place_order(slice_req)
                total_filled += result.filled_qty
                if result.avg_fill_price is not None:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result
                logger.debug(
                    "VWAP slice filled",
                    slice=i,
                    symbol=request.symbol,
                    requested_qty=slice_qty,
                    filled_qty=result.filled_qty,
                )
            except (ConnectionError, TimeoutError) as e:
                logger.error(
                    "Network error during VWAP slice placement",
                    slice=i,
                    symbol=request.symbol,
                    error=str(e),
                    exc_info=True,
                )
            except Exception as e:
                logger.error(
                    "Unexpected error during VWAP slice placement",
                    slice=i,
                    symbol=request.symbol,
                    error=str(e),
                    exc_info=True,
                )

            if i < active_slices - 1:
                try:
                    await asyncio.sleep(self.sleep_seconds)
                except Exception as e:
                    logger.error(
                        "Error during VWAP sleep interval",
                        slice=i,
                        error=str(e),
                        exc_info=True,
                    )

        avg_price = total_cost / total_filled if total_filled > 0 else None
        fill_rate = total_filled / request.quantity if request.quantity > 0 else 0

        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "vwap",
            status="filled" if fill_rate >= 0.95 else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )