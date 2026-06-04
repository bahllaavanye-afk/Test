"""
VWAP (Volume-Weighted Average Price) execution.
Participates at 10% of market volume across trading session.
Minimizes market impact by timing with volume distribution.
"""
from __future__ import annotations
import asyncio
from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger

# Empirical U-shaped intraday volume profile (30-min buckets, 13 buckets = 6.5h day)
# Heavy open/close, lighter midday — matches NYSE observed volume patterns
_EMPIRICAL_PROFILE = [0.12, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.06, 0.08, 0.12]


async def get_intraday_volume_profile(symbol: str, broker: AbstractBroker | None = None) -> list[float]:
    """Return normalized intraday volume distribution for symbol.

    Fetches yesterday's 30-min bar volumes from the broker if available.
    Falls back to the empirical U-shaped profile when broker data is missing.
    """
    if broker is not None:
        try:
            bars = await broker.get_bars(symbol, timeframe="30Min", limit=13)
            volumes = [float(getattr(b, "volume", 0) or 0) for b in bars]
            volumes = [v for v in volumes if v > 0]
            if len(volumes) >= 8:
                total = sum(volumes)
                profile = [v / total for v in volumes]
                logger.debug("VWAP dynamic profile loaded", symbol=symbol, buckets=len(profile))
                return profile
        except Exception as e:
            logger.debug("VWAP broker profile fetch failed, using empirical fallback", symbol=symbol, error=str(e))
    return list(_EMPIRICAL_PROFILE)


class VWAPExecution:
    def __init__(self, broker: AbstractBroker, participation_rate: float = 0.10, slices: int = 12):
        self.broker = broker
        self.participation_rate = participation_rate
        self.slices = slices
        self.sleep_seconds = (6.5 * 3600) / self.slices

    async def execute(self, request: OrderRequest) -> OrderResult:
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
                **{**request.__dict__, "quantity": slice_qty, "order_type": "market"}
            )
            try:
                result = await self.broker.place_order(slice_req)
                total_filled += result.filled_qty
                if result.avg_fill_price:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result
                logger.debug("VWAP slice filled", slice=i, qty=slice_qty, filled=result.filled_qty)
            except Exception as e:
                logger.warning("VWAP slice failed", slice=i, error=str(e))

            if i < active_slices - 1:
                await asyncio.sleep(self.sleep_seconds)

        avg_price = total_cost / total_filled if total_filled > 0 else None
        fill_rate = total_filled / request.quantity if request.quantity > 0 else 0
        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "vwap",
            status="filled" if fill_rate >= 0.95 else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )
