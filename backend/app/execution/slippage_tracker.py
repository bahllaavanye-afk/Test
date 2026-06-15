"""Tracks realized slippage vs expected fill price per order.

Item 5: Extended with Implementation Shortfall (IS) measurement.
IS = (fill_price - arrival_price) / arrival_price * 10000
where arrival_price is the mid-price when the order was first submitted.
"""
import uuid
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.brokers.base import OrderRequest, OrderResult
from app.models.slippage import SlippageRecord
from app.utils.logging import logger


class SlippageTracker:
    def __init__(self, db: AsyncSession | None = None):
        self.db = db
        self._signal_prices: dict[str, float] = {}
        # Item 5: track arrival prices (mid-price at order submission) and submit times
        self._arrival_prices: dict[str, float] = {}
        self._submit_times: dict[str, datetime] = {}

    async def record_signal_price(self, request: OrderRequest, signal_price: float) -> None:
        key = f"{request.account_id}:{request.symbol}"
        self._signal_prices[key] = signal_price

    async def record_arrival_price(self, request: OrderRequest, arrival_price: float) -> None:
        """Record mid-price at order submission time (for IS calculation)."""
        key = f"{request.account_id}:{request.symbol}"
        self._arrival_prices[key] = arrival_price
        self._submit_times[key] = datetime.now(UTC)

    async def record_fill(
        self,
        request: OrderRequest,
        result: OrderResult,
        period_vwap: float | None = None,
    ) -> None:
        if not result.avg_fill_price:
            return
        key = f"{request.account_id}:{request.symbol}"
        signal_price = self._signal_prices.pop(key, None)

        # Item 5: IS metrics
        arrival_price = self._arrival_prices.pop(key, None)
        submit_time = self._submit_times.pop(key, None)
        fill_time = datetime.now(UTC)
        execution_duration_seconds: float | None = None
        if submit_time is not None:
            execution_duration_seconds = (fill_time - submit_time).total_seconds()

        is_cost_bps: float | None = None
        if arrival_price and arrival_price > 0:
            if request.side == "buy":
                is_cost_bps = (result.avg_fill_price - arrival_price) / arrival_price * 10_000
            else:
                is_cost_bps = (arrival_price - result.avg_fill_price) / arrival_price * 10_000

        vwap_shortfall_bps: float | None = None
        if period_vwap and period_vwap > 0:
            if request.side == "buy":
                vwap_shortfall_bps = (result.avg_fill_price - period_vwap) / period_vwap * 10_000
            else:
                vwap_shortfall_bps = (period_vwap - result.avg_fill_price) / period_vwap * 10_000

        if signal_price and result.avg_fill_price:
            if request.side == "buy":
                slippage_bps = (result.avg_fill_price - signal_price) / signal_price * 10000
            else:
                slippage_bps = (signal_price - result.avg_fill_price) / signal_price * 10000

            logger.info(
                "Slippage recorded",
                symbol=request.symbol,
                expected=signal_price,
                fill=result.avg_fill_price,
                slippage_bps=round(slippage_bps, 2),
                is_cost_bps=round(is_cost_bps, 2) if is_cost_bps is not None else None,
                vwap_shortfall_bps=round(vwap_shortfall_bps, 2) if vwap_shortfall_bps is not None else None,
                duration_sec=round(execution_duration_seconds, 1) if execution_duration_seconds is not None else None,
                algo=request.execution_algo,
            )

            from app.notifications.slack import slack
            from app.notifications.tracker import tracker
            tracker.record("order_filled", "order",
                            f"{request.symbol} {request.side} filled @ {result.avg_fill_price}",
                            slippage_bps=round(slippage_bps, 2), algo=request.execution_algo)
            await slack.notify_order_filled(
                request.symbol, request.side, request.quantity,
                result.avg_fill_price, slippage_bps=round(slippage_bps, 2),
                algo=request.execution_algo,
            )

            if self.db:
                record = SlippageRecord(
                    id=str(uuid.uuid4()),
                    order_id=result.broker_order_id,
                    signal_price=signal_price,
                    expected_price=signal_price,
                    fill_price=result.avg_fill_price,
                    slippage_bps=slippage_bps,
                    execution_algo=request.execution_algo,
                    created_at=datetime.now(UTC),
                    # Item 5: IS fields
                    arrival_price=arrival_price,
                    is_cost_bps=is_cost_bps,
                    vwap_shortfall_bps=vwap_shortfall_bps,
                    period_vwap=period_vwap,
                    execution_duration_seconds=execution_duration_seconds,
                )
                self.db.add(record)
                await self.db.commit()

    async def get_execution_quality_stats(self, algo: str, days: int = 30) -> dict:
        """
        Item 5: Returns aggregated execution quality metrics for a given algo.

        Returns:
            dict with avg_is_bps, avg_slippage_bps, avg_vwap_shortfall_bps,
            avg_duration_seconds, num_fills, p95_is_bps.
        Raises RuntimeError if DB is not available.
        """
        if self.db is None:
            raise RuntimeError("DB session required for execution quality stats")

        from datetime import timedelta
        cutoff = datetime.now(UTC) - timedelta(days=days)

        stmt = (
            select(SlippageRecord)
            .where(SlippageRecord.execution_algo == algo)
            .where(SlippageRecord.created_at >= cutoff)
        )
        result = await self.db.execute(stmt)
        records = result.scalars().all()

        if not records:
            return {
                "algo": algo,
                "avg_is_bps": 0.0,
                "avg_slippage_bps": 0.0,
                "avg_vwap_shortfall_bps": 0.0,
                "avg_duration_seconds": 0.0,
                "num_fills": 0,
                "p95_is_bps": 0.0,
            }

        is_costs = [float(r.is_cost_bps) for r in records if r.is_cost_bps is not None]
        slippages = [float(r.slippage_bps) for r in records if r.slippage_bps is not None]
        vwap_shorts = [float(r.vwap_shortfall_bps) for r in records if r.vwap_shortfall_bps is not None]
        durations = [float(r.execution_duration_seconds) for r in records if r.execution_duration_seconds is not None]

        return {
            "algo": algo,
            "avg_is_bps": float(np.mean(is_costs)) if is_costs else 0.0,
            "avg_slippage_bps": float(np.mean(slippages)) if slippages else 0.0,
            "avg_vwap_shortfall_bps": float(np.mean(vwap_shorts)) if vwap_shorts else 0.0,
            "avg_duration_seconds": float(np.mean(durations)) if durations else 0.0,
            "num_fills": len(records),
            "p95_is_bps": float(np.percentile(is_costs, 95)) if is_costs else 0.0,
        }
