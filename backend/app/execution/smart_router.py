"""
Smart Order Router — selects the best execution algorithm based on order characteristics.

The router evaluates order size, asset type, urgency, and any explicit algorithm
override to decide among several execution strategies:

* Very large orders (≥ $100 k) use a reinforcement‑learning agent if available,
  otherwise fall back to TWAP.
* Large orders (≥ $100 k) – TWAP over 30 min.
* Mid‑size orders ($5 k – $100 k) – Almgren‑Chriss optimal trajectory.
* Limit orders with a limit price – Limit‑first (post limit, fallback to market).
* All other orders – market order.

All orders pass through an optional ``risk_manager`` before execution.  The
router records signal and fill information with ``SlippageTracker`` when
provided.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from app.brokers.base import OrderRequest, OrderResult, AbstractBroker
from app.execution.limit_first import LimitFirstExecution
from app.execution.twap import TWAPExecution
from app.execution.slippage_tracker import SlippageTracker
from app.execution.almgren_chriss import AlmgrenChriss
from app.utils.logging import logger

try:
    from app.execution.rl_exec import RLExecution, get_rl_agent

    _RL_EXEC_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 – optional dependency
    _RL_EXEC_AVAILABLE = False
    logger.debug(
        "RL execution unavailable — falling back to rule router (%s)",
        exc,
    )


class SmartOrderRouter:
    """
    Routes an :class:`~app.brokers.base.OrderRequest` to the most appropriate
    execution algorithm.

    Parameters
    ----------
    broker: AbstractBroker
        The broker implementation used to place orders.
    slippage_tracker: Optional[SlippageTracker]
        Tracker that records signal and fill information for post‑trade analysis.
    risk_manager: Any, optional
        An object exposing ``await check_order(request)`` returning an object
        with ``allowed`` (bool), ``reason`` (str) and optional ``adjusted_quantity``.
        If ``None`` the router skips risk checks.
    """

    def __init__(
        self,
        broker: AbstractBroker,
        slippage_tracker: Optional[SlippageTracker] = None,
        risk_manager: Any = None,
    ) -> None:
        self.broker = broker
        self.slippage_tracker = slippage_tracker
        self.risk_manager = risk_manager

    async def execute(
        self,
        request: OrderRequest,
        signal_price: float | None = None,
    ) -> OrderResult | None:
        """
        Execute an order using the algorithm selected by ``_select_algorithm``.

        The method first asks the optional ``risk_manager`` to validate the order.
        If the order is blocked, ``None`` is returned and a warning is logged.
        Otherwise the request is routed, the chosen algorithm is attached to the
        request, and the result (or ``None`` on failure) is returned.

        Parameters
        ----------
        request: OrderRequest
            The order to be executed.
        signal_price: float | None
            Optional price associated with the triggering signal; used for
            slippage tracking.

        Returns
        -------
        OrderResult | None
            The execution result, or ``None`` if the order was blocked by risk
            management.
        """
        # ── Risk gate ────────────────────────────────────────────────────────
        if self.risk_manager is not None:
            decision = await self.risk_manager.check_order(request)
            if not decision.allowed:
                logger.warning(
                    "Order blocked by risk manager",
                    symbol=request.symbol,
                    reason=decision.reason,
                )
                return None
            if decision.adjusted_quantity is not None:
                request.quantity = decision.adjusted_quantity

        algo = self._select_algorithm(request)
        request.execution_algo = algo

        # Record signal price for slippage tracking
        if signal_price is not None and self.slippage_tracker is not None:
            await self.slippage_tracker.record_signal_price(request, signal_price)

        if algo == "almgren_chriss":
            result = await self._execute_almgren_chriss(request)
        elif algo == "twap":
            result = await TWAPExecution(
                self.broker, slices=10, duration_minutes=30
            ).execute(request)
        elif algo == "limit_first":
            result = await LimitFirstExecution(
                self.broker, offset_bps=5, fallback_seconds=30
            ).execute(request)
        elif algo == "rl_exec" and _RL_EXEC_AVAILABLE:
            fills = await RLExecution(
                self.broker, agent=get_rl_agent()
            ).execute(request, signal_price)
            # Aggregate fills into a single OrderResult for compatibility
            if fills:
                total_qty = sum(f["qty"] for f in fills)
                avg_price = sum(f["qty"] * f["price"] for f in fills) / max(
                    total_qty, 1e-9
                )
                result = OrderResult(
                    order_id=f"rl_{request.symbol}",
                    symbol=request.symbol,
                    status="filled",
                    filled_qty=total_qty,
                    avg_fill_price=avg_price,
                )
            else:
                result = None
        else:
            result = await self.broker.place_order(request)

        if self.slippage_tracker is not None:
            await self.slippage_tracker.record_fill(request, result)

        return result

    def _select_algorithm(self, request: OrderRequest) -> str:
        """
        Choose the execution algorithm based on order characteristics.

        The selection order is:

        1. Explicit ``execution_algo`` override on the request.
        2. Very large orders (≥ $100 k) – RL agent if available, else TWAP.
        3. Large orders (≥ $100 k) – TWAP.
        4. Mid‑size orders ($5 k – $100 k) – Almgren‑Chriss.
        5. Limit orders with a limit price – Limit‑first.
        6. Default – market order.

        Returns
        -------
        str
            The identifier of the chosen algorithm.
        """
        # Resolve a reference price for USD‑valuation.  Preference:
        # limit_price → stop_price → signal_price (metadata) → $50 fallback.
        ref_price = (
            request.limit_price
            or request.stop_price
            or (asdict(request).get("metadata") or {}).get("signal_price")
            or 50.0
        )
        estimated_usd = request.quantity * ref_price

        if request.execution_algo and request.execution_algo not in ("auto", ""):
            return request.execution_algo  # explicit user/strategy override
        if estimated_usd >= 100_000 and _RL_EXEC_AVAILABLE:
            return "rl_exec"  # RL agent for very large orders (better than TWAP)
        if estimated_usd >= 100_000:
            return "twap"
        if 5_000 <= estimated_usd < 100_000:
            return "almgren_chriss"  # optimal IS minimisation for mid‑size orders
        if request.order_type == "limit" and request.limit_price:
            return "limit_first"
        return "market"

    async def _execute_almgren_chriss(self, request: OrderRequest) -> OrderResult:
        """
        Execute the order using the Almgren‑Chriss optimal trajectory.

        The algorithm splits the total quantity into a series of market slices
        (as limit orders are not supported by the underlying broker).  Each slice
        is submitted sequentially with a pause that spreads the execution over the
        target duration.

        Returns
        -------
        OrderResult
            Aggregated result of all slices, containing the final filled quantity
            and volume‑weighted average price.
        """
        import asyncio

        # Estimate volatility (sigma) from metadata if available; default 2 %.
        sigma = (
            float(asdict(request).get("metadata", {}).get("sigma", 0.02))
            if hasattr(request, "__dict__")
            else 0.02
        )
        ac = AlmgrenChriss(sigma=sigma)
        n_slices = 10
        duration_minutes = 20
        trades = ac.optimal_trajectory(request.quantity, duration_minutes, n_slices)
        sleep_secs = (duration_minutes * 60) / n_slices

        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None
        consecutive_failures = 0

        for i, slice_qty in enumerate(trades):
            if slice_qty < 1e-6:
                continue
            # Use market slices – adding "limit" without a price causes broker rejection.
            slice_req = OrderRequest(
                **{
                    **asdict(request),
                    "quantity": float(slice_qty),
                    "order_type": "market",
                    "limit_price": None,
                }
            )
            try:
                result = await self.broker.place_order(slice_req)
                total_filled += result.filled_qty
                if result.avg_fill_price:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result
                consecutive_failures = 0
                logger.debug(
                    "AC slice executed",
                    symbol=request.symbol,
                    slice=i + 1,
                    n_slices=n_slices,
                    qty=round(slice_qty, 4),
                )
            except Exception as e:  # noqa: BLE001
                consecutive_failures += 1
                logger.warning(
                    "AC slice failed",
                    symbol=request.symbol,
                    slice=i + 1,
                    error=str(e),
                )
                if consecutive_failures >= 3:
                    logger.error(
                        "AC execution aborting after consecutive failures",
                        symbol=request.symbol,
                    )
                    break

            if i < len(trades) - 1:
                await asyncio.sleep(sleep_secs)

        avg_price = total_cost / total_filled if total_filled > 0 else None
        cost_info = ac.expected_cost(request.quantity, duration_minutes, n_slices)
        logger.info(
            "AC execution complete",
            symbol=request.symbol,
            filled=round(total_filled, 4),
            avg_price=avg_price,
            expected_total_cost=round(cost_info["total"], 6),
        )
        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "ac_exec",
            symbol=request.symbol,
            status="filled"
            if total_filled >= request.quantity * 0.99
            else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )