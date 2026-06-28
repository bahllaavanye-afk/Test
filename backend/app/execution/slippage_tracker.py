"""Tracks realized slippage versus expected fill price per order.

This module records signal prices, arrival (mid) prices, and fill information
to compute several execution quality metrics, including:

* Slippage (bps) relative to the signal price.
* Implementation Shortfall (IS) (bps) relative to the arrival price.
* VWAP short‑fall (bps) relative to a period VWAP, if provided.
* Execution duration (seconds).

The data is persisted to the database via the ``SlippageRecord`` ORM model
and can be aggregated with :meth:`SlippageTracker.get_execution_quality_stats`.

Item 5: Extended with Implementation Shortfall (IS) measurement.
IS = (fill_price - arrival_price) / arrival_price * 10_000
where ``arrival_price`` is the mid‑price when the order was first submitted.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.brokers.base import OrderRequest, OrderResult
from app.models.slippage import SlippageRecord
from app.utils.logging import logger


class SlippageTracker:
    """Collects and persists per‑order execution metrics.

    Parameters
    ----------
    db : Optional[AsyncSession]
        An asynchronous SQLAlchemy session used to store ``SlippageRecord`` rows.
        If ``None``, database persistence is skipped.
    """

    def __init__(self, db: Optional[AsyncSession] = None) -> None:
        self.db = db
        self._signal_prices: Dict[str, float] = {}
        # Item 5: track arrival prices (mid‑price at order submission) and submit times
        self._arrival_prices: Dict[str, float] = {}
        self._submit_times: Dict[str, datetime] = {}

    async def record_signal_price(self, request: OrderRequest, signal_price: float) -> None:
        """Store the expected fill price (signal price) for a given order.

        The key is a composite of ``account_id`` and ``symbol`` to allow
        correlation with a later fill event.

        Parameters
        ----------
        request : OrderRequest
            The original order request.
        signal_price : float
            Expected fill price derived from the trading signal.
        """
        key = f"{request.account_id}:{request.symbol}"
        self._signal_prices[key] = signal_price

    async def record_arrival_price(self, request: OrderRequest, arrival_price: float) -> None:
        """Record the mid‑price at order submission time (used for IS calculation).

        Parameters
        ----------
        request : OrderRequest
            The original order request.
        arrival_price : float
            The mid‑price observed when the order was first submitted.
        """
        key = f"{request.account_id}:{request.symbol}"
        self._arrival_prices[key] = arrival_price
        self._submit_times[key] = datetime.now(UTC)

    async def record_fill(
        self,
        request: OrderRequest,
        result: OrderResult,
        period_vwap: Optional[float] = None,
    ) -> None:
        """Persist a filled order and compute execution metrics.

        The method calculates slippage, implementation shortfall, VWAP short‑fall,
        and execution duration. It logs the results, sends optional Slack
        notifications, and stores a ``SlippageRecord`` in the database if a session
        is available.

        Parameters
        ----------
        request : OrderRequest
            The original order request.
        result : OrderResult
            The broker‑provided fill result. ``avg_fill_price`` must be populated.
        period_vwap : Optional[float]
            VWAP for the period covering the order, used for VWAP short‑fall.
        """
        if not result.avg_fill_price:
            return

        key = f"{request.account_id}:{request.symbol}"
        signal_price = self._signal_prices.pop(key, None)

        # Item 5: IS metrics
        arrival_price = self._arrival_prices.pop(key, None)
        submit_time = self._submit_times.pop(key, None)
        fill_time = datetime.now(UTC)
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
                vwap_shortfall_bps=round(vwap_shortfall_bps, 2) if vwap_shortfall_bps is not None else None,
                duration_sec=round(execution_duration_seconds, 1) if execution_duration_seconds is not None else None,
                algo=request.execution_algo,
            )

            try:
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
            except Exception as exc:
                logger.error(
                    "Failed to send fill notification",
                    symbol=request.symbol,
                    side=request.side,
                    exception=str(exc),
                )

            if self.db:
                try:
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
                except SQLAlchemyError as db_err:
                    logger.error(
                        "Database error while recording slippage",
                        order_id=result.broker_order_id,
                        exception=str(db_err),
                    )
                except Exception as exc:
                    logger.error(
                        "Unexpected error while recording slippage",
                        order_id=result.broker_order_id,
                        exception=str(exc),
                    )

    async def get_execution_quality_stats(self, algo: str, days: int = 30) -> Dict[str, Any]:
        """Aggregate execution quality metrics for a specific algorithm.

        The method fetches ``SlippageRecord`` rows for the given ``algo`` within
        the last ``days`` days and computes average slippage, IS, VWAP short‑fall,
        execution duration, fill count, and the 95th percentile of IS.

        Parameters
        ----------
        algo : str
            The execution algorithm identifier.
        days : int, default ``30``
            Look‑back window in days.

        Returns
        -------
        dict
            A mapping containing:

            * ``algo`` (str) – algorithm identifier.
            * ``avg_is_bps`` (float) – mean implementation shortfall.
            * ``avg_slippage_bps`` (float) – mean slippage.
            * ``avg_vwap_shortfall_bps`` (float) – mean VWAP short‑fall.
            * ``avg_duration_seconds`` (float) – mean execution duration.
            * ``num_fills`` (int) – total number of fills.
            * ``p95_is_bps`` (float) – 95th percentile of IS values.

        Raises
        ------
        RuntimeError
            If a database session is not configured or a DB error occurs.
        """
        if self.db is None:
            raise RuntimeError("DB session required for execution quality stats")

        cutoff = datetime.now(UTC) - timedelta(days=days)

        stmt = (
            select(SlippageRecord)
            .where(SlippageRecord.execution_algo == algo)
            .where(SlippageRecord.created_at >= cutoff)
        )
        try:
            result = await self.db.execute(stmt)
        except SQLAlchemyError as db_err:
            logger.error(
                "Database error while fetching execution quality stats",
                algo=algo,
                days=days,
                exception=str(db_err),
            )
            raise RuntimeError("Failed to retrieve execution quality stats") from db_err
        except Exception as exc:
            logger.error(
                "Unexpected error while fetching execution quality stats",
                algo=algo,
                days=days,
                exception=str(exc),
            )
            raise RuntimeError("Failed to retrieve execution quality stats") from exc

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

        is_vals: List[float] = []
        slippage_vals: List[float] = []
        vwap_vals: List[float] = []
        duration_vals: List[float] = []

        for rec in records:
            if rec.is_cost_bps is not None:
                is_vals.append(rec.is_cost_bps)
            if rec.slippage_bps is not None:
                slippage_vals.append(rec.slippage_bps)
            if rec.vwap_shortfall_bps is not None:
                vwap_vals.append(rec.vwap_shortfall_bps)
            if rec.execution_duration_seconds is not None:
                duration_vals.append(rec.execution_duration_seconds)

        num_fills = len(records)

        avg_is = float(np.mean(is_vals)) if is_vals else 0.0
        avg_slippage = float(np.mean(slippage_vals)) if slippage_vals else 0.0
        avg_vwap = float(np.mean(vwap_vals)) if vwap_vals else 0.0
        avg_duration = float(np.mean(duration_vals)) if duration_vals else 0.0
        p95_is = float(np.percentile(is_vals, 95)) if is_vals else 0.0

        return {
            "algo": algo,
            "avg_is_bps": avg_is,
            "avg_slippage_bps": avg_slippage,
            "avg_vwap_shortfall_bps": avg_vwap,
            "avg_duration_seconds": avg_duration,
            "num_fills": num_fills,
            "p95_is_bps": p95_is,
        }