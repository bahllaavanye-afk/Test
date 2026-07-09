"""Execution quality tracking utilities.

This module provides a :class:`SlippageTracker` that records signal and arrival
prices for orders, computes various execution quality metrics such as slippage,
implementation shortfall (IS), and VWAP shortfall, and persists the results to
the database.  The tracker also forwards notifications to Slack and a generic
tracker service.
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.brokers.base import OrderRequest, OrderResult
from app.models.slippage import SlippageRecord
from app.utils.logging import logger


class SlippageTracker:
    """Tracks realized slippage versus expected fill price per order.

    The tracker records:

    * The *signal price* – the price at which the trading algorithm signaled an
      order.
    * The *arrival price* – the mid‑price when the order was first submitted,
      used for implementation shortfall (IS) calculation.
    * The time of order submission, to compute execution duration.

    When a fill is reported, the tracker calculates slippage, IS, VWAP shortfall,
    and execution duration, logs the information, sends notifications, and
    persists a :class:`SlippageRecord` if a database session is provided.
    """

    def __init__(self, db: Optional[AsyncSession] = None) -> None:
        """Create a new :class:`SlippageTracker`.

        Args:
            db: Optional asynchronous SQLAlchemy session used to store
                :class:`SlippageRecord` instances. If ``None`` the tracker will
                operate without persisting data.
        """
        self.db = db
        self._signal_prices: Dict[str, float] = {}
        self._arrival_prices: Dict[str, float] = {}
        self._submit_times: Dict[str, datetime] = {}

    async def record_signal_price(
        self, request: OrderRequest, signal_price: float
    ) -> None:
        """Store the expected price for a given order.

        Args:
            request: The original order request containing ``account_id`` and
                ``symbol``.
            signal_price: The price that the algorithm expects for the order.
        """
        key = f"{request.account_id}:{request.symbol}"
        self._signal_prices[key] = signal_price

    async def record_arrival_price(
        self, request: OrderRequest, arrival_price: float
    ) -> None:
        """Record the mid‑price at order submission time (for IS calculation).

        Args:
            request: The original order request.
            arrival_price: The mid‑price observed when the order was submitted.
        """
        key = f"{request.account_id}:{request.symbol}"
        self._arrival_prices[key] = arrival_price
        self._submit_times[key] = datetime.now(timezone.utc)

    async def record_fill(
        self,
        request: OrderRequest,
        result: OrderResult,
        period_vwap: Optional[float] = None,
    ) -> None:
        """Process a filled order and compute execution quality metrics.

        The method calculates slippage, implementation shortfall, VWAP shortfall,
        and execution duration (if submission time is known).  It logs the
        results, sends notifications, and stores a :class:`SlippageRecord` when
        a database session is available.

        Args:
            request: The original order request.
            result: The result returned by the broker, containing fill price and
                order identifier.
            period_vwap: Optional VWAP price for the period in which the order
                was executed, used to compute VWAP shortfall.
        """
        if not result.avg_fill_price:
            return

        key = f"{request.account_id}:{request.symbol}"
        signal_price = self._signal_prices.pop(key, None)

        # Implementation shortfall (IS) metrics
        arrival_price = self._arrival_prices.pop(key, None)
        submit_time = self._submit_times.pop(key, None)
        fill_time = datetime.now(timezone.utc)
        execution_duration_seconds: Optional[float] = None
        if submit_time is not None:
            execution_duration_seconds = (fill_time - submit_time).total_seconds()

        is_cost_bps: Optional[float] = None
        if arrival_price and arrival_price > 0:
            if request.side == "buy":
                is_cost_bps = (result.avg_fill_price - arrival_price) / arrival_price * 10_000
            else:
                is_cost_bps = (arrival_price - result.avg_fill_price) / arrival_price * 10_000

        vwap_shortfall_bps: Optional[float] = None
        if period_vwap and period_vwap > 0:
            if request.side == "buy":
                vwap_shortfall_bps = (result.avg_fill_price - period_vwap) / period_vwap * 10_000
            else:
                vwap_shortfall_bps = (period_vwap - result.avg_fill_price) / period_vwap * 10_000

        if signal_price and result.avg_fill_price:
            if request.side == "buy":
                slippage_bps = (result.avg_fill_price - signal_price) / signal_price * 10_000
            else:
                slippage_bps = (signal_price - result.avg_fill_price) / signal_price * 10_000

            logger.info(
                "Slippage recorded",
                symbol=request.symbol,
                expected=signal_price,
                fill=result.avg_fill_price,
                slippage_bps=round(slippage_bps, 2),
                is_cost_bps=round(is_cost_bps, 2) if is_cost_bps is not None else None,
                vwap_shortfall_bps=round(vwap_shortfall_bps, 2)
                if vwap_shortfall_bps is not None
                else None,
                duration_sec=round(execution_duration_seconds, 1)
                if execution_duration_seconds is not None
                else None,
                algo=request.execution_algo,
            )

            from app.notifications.slack import slack
            from app.notifications.tracker import tracker

            tracker.record(
                "order_filled",
                "order",
                f"{request.symbol} {request.side} filled @ {result.avg_fill_price}",
                slippage_bps=round(slippage_bps, 2),
                algo=request.execution_algo,
            )
            await slack.notify_order_filled(
                request.symbol,
                request.side,
                request.quantity,
                result.avg_fill_price,
                slippage_bps=round(slippage_bps, 2),
                algo=request.execution_algo,
            )

            # Duplicate notification block retained for backward compatibility
            from app.notifications.slack import slack
            from app.notifications.tracker import tracker

            tracker.record(
                "order_filled",
                "order",
                f"{request.symbol} {request.side} filled @ {result.avg_fill_price}",
                slippage_bps=round(slippage_bps, 2),
                algo=request.execution_algo,
            )
            await slack.notify_order_filled(
                request.symbol,
                request.side,
                request.quantity,
                result.avg_fill_price,
                slippage_bps=round(slippage_bps, 2),
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
                    created_at=datetime.now(timezone.utc),
                    arrival_price=arrival_price,
                    is_cost_bps=is_cost_bps,
                    vwap_shortfall_bps=vwap_shortfall_bps,
                    period_vwap=period_vwap,
                    execution_duration_seconds=execution_duration_seconds,
                )
                self.db.add(record)
                await self.db.commit()

    async def get_execution_quality_stats(
        self, algo: str, days: int = 30
    ) -> Dict[str, Any]:
        """Return aggregated execution quality metrics for a given algorithm.

        The method queries the persisted :class:`SlippageRecord` entries for the
        specified ``algo`` within the past ``days`` days and computes average and
        percentile statistics.

        Args:
            algo: The execution algorithm identifier to filter records.
            days: Number of days in the past to include in the aggregation.

        Returns:
            A dictionary containing:

            * ``algo`` – the algorithm identifier.
            * ``avg_is_bps`` – average implementation shortfall in basis points.
            * ``avg_slippage_bps`` – average slippage in basis points.
            * ``avg_vwap_shortfall_bps`` – average VWAP shortfall in basis points.
            * ``avg_duration_seconds`` – average execution duration in seconds.
            * ``num_fills`` – total number of fills considered.
            * ``p95_is_bps`` – 95th percentile of IS in basis points.

        Raises:
            RuntimeError: If the tracker was instantiated without a database
                session.
        """
        if self.db is None:
            raise RuntimeError("DB session required for execution quality stats")

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        stmt = (
            select(SlippageRecord)
            .where(SlippageRecord.execution_algo == algo)
            .where(SlippageRecord.created_at >= cutoff)
        )
        result = await self.db.execute(stmt)
        records: List[SlippageRecord] = result.scalars().all()

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
        vwap_shorts = [
            float(r.vwap_shortfall_bps) for r in records if r.vwap_shortfall_bps is not None
        ]
        durations = [
            float(r.execution_duration_seconds)
            for r in records
            if r.execution_duration_seconds is not None
        ]

        return {
            "algo": algo,
            "avg_is_bps": float(np.mean(is_costs)) if is_costs else 0.0,
            "avg_slippage_bps": float(np.mean(slippages)) if slippages else 0.0,
            "avg_vwap_shortfall_bps": float(np.mean(vwap_shorts)) if vwap_shorts else 0.0,
            "avg_duration_seconds": float(np.mean(durations)) if durations else 0.0,
            "num_fills": len(records),
            "p95_is_bps": float(np.percentile(is_costs, 95)) if is_costs else 0.0,
        }