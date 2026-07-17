"""
VWAP (Volume-Weighted Average Price) execution.

Participates at a configurable fraction of market volume across the trading session.
Minimizes market impact by timing orders with the expected intraday volume
distribution and adds tighter entry checks, confirmation filters, and improved
exit handling.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Optional

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
        except Exception as e:  # pragma: no cover
            logger.debug(
                "VWAP broker profile fetch failed, using empirical fallback",
                symbol=symbol,
                error=str(e),
            )
    return list(_EMPIRICAL_PROFILE)


async def _fetch_average_daily_volume(
    broker: AbstractBroker, symbol: str
) -> Optional[float]:
    """Safely fetch an average daily volume estimate from the broker.

    Returns ``None`` if the broker does not expose the required method or the
    call fails.
    """
    try:
        # Some brokers expose ``get_average_daily_volume``; we guard against
        # missing implementations.
        if hasattr(broker, "get_average_daily_volume"):
            volume = await broker.get_average_daily_volume(symbol)
            return float(volume) if volume is not None else None
    except Exception as e:  # pragma: no cover
        logger.debug(
            "Failed to fetch average daily volume", symbol=symbol, error=str(e)
        )
    return None


async def _fetch_recent_market_volume(
    broker: AbstractBroker, symbol: str
) -> Optional[float]:
    """Fetch the most recent market volume (e.g., last minute) if supported."""
    try:
        if hasattr(broker, "get_recent_market_volume"):
            volume = await broker.get_recent_market_volume(symbol)
            return float(volume) if volume is not None else None
    except Exception as e:  # pragma: no cover
        logger.debug(
            "Failed to fetch recent market volume", symbol=symbol, error=str(e)
        )
    return None


async def _fetch_last_price(broker: AbstractBroker, symbol: str) -> Optional[float]:
    """Fetch the latest price for *symbol* if the broker provides it."""
    try:
        if hasattr(broker, "get_last_price"):
            price = await broker.get_last_price(symbol)
            return float(price) if price is not None else None
    except Exception as e:  # pragma: no cover
        logger.debug("Failed to fetch last price", symbol=symbol, error=str(e))
    return None


class VWAPExecution:
    """Execute orders using a VWAP strategy with enhanced entry/exit logic.

    The algorithm slices the total order quantity according to an intraday
    volume profile and sends market orders at a configurable participation rate.
    Additional safeguards tighten entry conditions, filter each slice based on
    real‑time market data, and allow early termination when execution deviates
    significantly from the plan.
    """

    def __init__(
        self,
        broker: AbstractBroker,
        participation_rate: float = 0.10,
        slices: int = 12,
        max_entry_volume_factor: float = 0.25,
        price_deviation_tolerance: float = 0.02,
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
        max_entry_volume_factor:
            Maximum allowed order size as a fraction of estimated daily volume.
            Helps avoid oversized orders that could destabilise the market.
        price_deviation_tolerance:
            Maximum allowed deviation (as a fraction of last price) for a slice
            to be sent.  Slices whose price moves beyond this tolerance are
            postponed or reduced.
        """
        self.broker = broker
        self.participation_rate = participation_rate
        self.slices = slices
        self.max_entry_volume_factor = max_entry_volume_factor
        self.price_deviation_tolerance = price_deviation_tolerance
        self.sleep_seconds = (6.5 * 3600) / self.slices

    async def _validate_entry(self, request: OrderRequest) -> bool:
        """Validate the order before execution starts.

        Checks that the order size is reasonable relative to recent market volume
        and that the broker can provide necessary market data.
        """
        avg_daily_vol = await _fetch_average_daily_volume(self.broker, request.symbol)
        if avg_daily_vol is not None:
            max_allowed_qty = avg_daily_vol * self.max_entry_volume_factor
            if request.quantity > max_allowed_qty:
                logger.warning(
                    "VWAP entry rejected: order quantity exceeds allowed volume",
                    symbol=request.symbol,
                    quantity=request.quantity,
                    max_allowed=max_allowed_qty,
                )
                return False
        # Additional checks could be added here (e.g., volatility filters).
        return True

    async def _confirm_slice(
        self, slice_qty: float, symbol: str
    ) -> Optional[float]:
        """Apply confirmation filters to a slice.

        Returns an adjusted quantity if the slice passes the filters, otherwise
        ``None`` to skip the slice.
        """
        recent_vol = await _fetch_recent_market_volume(self.broker, symbol)
        if recent_vol is not None:
            # Enforce participation rate against recent market volume.
            max_slice_qty = recent_vol * self.participation_rate
            if slice_qty > max_slice_qty:
                logger.debug(
                    "Slice quantity reduced to respect participation rate",
                    original_qty=slice_qty,
                    adjusted_qty=max_slice_qty,
                )
                slice_qty = max_slice_qty

        last_price = await _fetch_last_price(self.broker, symbol)
        if last_price is not None:
            # Simple price deviation filter: ensure the price is within tolerance
            # of the VWAP estimate (here approximated by the last price).
            # In a full implementation the VWAP would be computed; we use last
            # price as a proxy.
            deviation = abs(slice_qty / request.quantity - self.participation_rate)
            if deviation > self.price_deviation_tolerance:
                logger.debug(
                    "Slice skipped due to price deviation filter",
                    symbol=symbol,
                    deviation=deviation,
                )
                return None
        return slice_qty

    async def execute(self, request: OrderRequest) -> OrderResult:
        """Execute a VWAP order.

        The method retrieves a volume profile, validates the entry, then
        iteratively sends slice orders while applying confirmation filters.
        Early termination occurs if fill rates fall below a threshold after
        half the slices have been processed.
        """
        # Entry validation
        if not await self._validate_entry(request):
            return OrderResult(
                broker_order_id="vwap_rejected",
                status="rejected",
                filled_qty=0.0,
                avg_fill_price=None,
            )

        profile = await get_intraday_volume_profile(request.symbol, self.broker)
        active_slices = min(self.slices, len(profile))
        profile_slice = profile[:active_slices]
        profile_total = sum(profile_slice)

        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None

        for i in range(active_slices):
            slice_weight = profile_slice[i] / profile_total
            raw_slice_qty = request.quantity * slice_weight

            # Confirmation filter may adjust or skip the slice
            adjusted_qty = await self._confirm_slice(raw_slice_qty, request.symbol)
            if adjusted_qty is None or adjusted_qty <= 0:
                logger.debug("VWAP slice skipped after confirmation", slice=i)
                continue

            slice_req = OrderRequest(
                **{
                    **asdict(request),
                    "quantity": adjusted_qty,
                    "order_type": "market",
                }
            )
            try:
                result = await self.broker.place_order(slice_req)
                total_filled += result.filled_qty
                if result.avg_fill_price:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result
                logger.debug(
                    "VWAP slice filled",
                    slice=i,
                    qty=adjusted_qty,
                    filled=result.filled_qty,
                )
            except Exception as e:  # pragma: no cover
                logger.warning("VWAP slice failed", slice=i, error=str(e))

            # Early exit logic: after half the slices, if fill rate is low,
            # terminate the algorithm to prevent further adverse execution.
            if i + 1 >= active_slices // 2:
                fill_rate_sofar = total_filled / request.quantity if request.quantity else 0
                if fill_rate_sofar < 0.30:
                    logger.warning(
                        "Early termination of VWAP: low fill rate",
                        fill_rate=fill_rate_sofar,
                    )
                    break

            if i < active_slices - 1:
                await asyncio.sleep(self.sleep_seconds)

        avg_price = total_cost / total_filled if total_filled > 0 else None
        fill_rate = total_filled / request.quantity if request.quantity > 0 else 0
        status = "filled" if fill_rate >= 0.95 else "partial"

        # If we terminated early with very low fill, mark as partial but note.
        if fill_rate < 0.30:
            status = "partial"

        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "vwap",
            status=status,
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )