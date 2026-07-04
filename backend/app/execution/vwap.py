"""
VWAP (Volume-Weighted Average Price) execution.

Participates at 10 % of market volume across the trading session.
Minimizes market impact by timing orders with the expected intraday volume
distribution.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
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
) -> list[float]:
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
    list[float]
        Normalised volume weights that sum to 1.0 (or the empirical profile if
        dynamic data could not be obtained).
    """
    if broker is not None:
        try:
            bars = await broker.get_bars(symbol, timeframe="30Min", limit=13)
            volumes = [float(getattr(b, "volume", 0) or 0) for b in bars]
            volumes = [v for v in volumes if v > 0]
            if len(volumes) >= 8:
                total = sum(volumes)
                profile = [v / total for v in volumes]
                logger.debug(
                    "VWAP dynamic profile loaded", symbol=symbol, buckets=len(profile)
                )
                return profile
        except Exception as e:
            logger.debug(
                "VWAP broker profile fetch failed, using empirical fallback",
                symbol=symbol,
                error=str(e),
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
        profile = await get_intraday_volume_profile(request.symbol, self.broker)
        slice_weights = self._compute_slice_weights(profile)
        total_filled, total_cost, last_result = await self._run_slices(
            request, slice_weights
        )
        avg_price = total_cost / total_filled if total_filled > 0 else None
        fill_rate = total_filled / request.quantity if request.quantity > 0 else 0
        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "vwap",
            status="filled" if fill_rate >= 0.95 else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )

    def _compute_slice_weights(self, profile: list[float]) -> list[float]:
        """Calculate normalized slice weights based on the intraday profile.

        The number of slices is limited by both the configured ``self.slices`` and
        the length of the provided profile.
        """
        active_slices = min(self.slices, len(profile))
        relevant_profile = profile[:active_slices]
        total = sum(relevant_profile)
        # Guard against division by zero (unlikely with a valid profile)
        if total == 0:
            return [0.0] * active_slices
        return [weight / total for weight in relevant_profile]

    async def _run_slices(
        self, request: OrderRequest, slice_weights: list[float]
    ) -> tuple[float, float, OrderResult | None]:
        """Iterate over slice weights, place orders and aggregate results.

        Returns
        -------
        total_filled: float
            Sum of filled quantities across all slices.
        total_cost: float
            Cumulative cost (price * quantity) for filled portions.
        last_result: OrderResult | None
            The most recent successful order result, used for broker_order_id.
        """
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None

        for i, weight in enumerate(slice_weights):
            slice_qty = request.quantity * weight
            slice_req = OrderRequest(
                **{**asdict(request), "quantity": slice_qty, "order_type": "market"}
            )
            try:
                result = await self.broker.place_order(slice_req)
                total_filled += result.filled_qty
                if result.avg_fill_price:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result
                logger.debug(
                    "VWAP slice filled", slice=i, qty=slice_qty, filled=result.filled_qty
                )
            except Exception as e:
                logger.warning("VWAP slice failed", slice=i, error=str(e))

            # Sleep between slices except after the final one
            if i < len(slice_weights) - 1:
                await asyncio.sleep(self.sleep_seconds)

        return total_filled, total_cost, last_result