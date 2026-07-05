"""
Smart Order Router — selects the best execution algorithm based on order characteristics.

The router evaluates an incoming `OrderRequest` and chooses an execution strategy
(e.g., TWAP, Almgren‑Chriss, limit‑first, RL‑based, or a simple market order) that
optimizes for minimal slippage while respecting risk constraints.

All orders are first checked by an optional `risk_manager`. If the risk manager
blocks the order, the router logs a warning and returns ``None``. Otherwise the
selected algorithm is executed and the result is optionally recorded by a
`SlippageTracker`.
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
except Exception:
    _RL_EXEC_AVAILABLE = False


class SmartOrderRouter:
    """Routes orders to the most appropriate execution algorithm.

    Parameters
    ----------
    broker : AbstractBroker
        Broker instance used to place orders.
    slippage_tracker : Optional[SlippageTracker]
        Tracker for recording signal and fill prices; can be ``None``.
    risk_manager : Optional[Any]
        Optional risk manager with an ``async check_order`` method. The exact
        interface is not enforced here to keep the router decoupled from a
        specific implementation.
    """

    def __init__(
        self,
        broker: AbstractBroker,
        slippage_tracker: Optional[SlippageTracker] = None,
        risk_manager: Optional[Any] = None,
    ) -> None:
        self.broker = broker
        self.slippage_tracker = slippage_tracker
        self.risk_manager = risk_manager

    async def execute(self, request: OrderRequest, signal_price: float | None = None) -> OrderResult | None:
        """Route an order to the optimal execution algorithm.

        The method performs the following steps:
        1. Checks the order with the risk manager (if configured).
        2. Selects an execution algorithm based on order size, type, and other
           characteristics.
        3. Executes the order using the chosen algorithm.
        4. Records signal and fill information with the slippage tracker (if
           configured).

        Parameters
        ----------
        request : OrderRequest
            The order to be executed.
        signal_price : float | None, optional
            Optional price associated with the originating trading signal.

        Returns
        -------
        OrderResult | None
            The execution result, or ``None`` if the order was blocked by the
            risk manager.
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
        if signal_price and self.slippage_tracker:
            await self.slippage_tracker.record_signal_price(request, signal_price)

        if algo == "almgren_chriss":
            result = await self._execute_almgren_chriss(request)
        elif algo == "twap":
            result = await TWAPExecution(self.broker, slices=10, duration_minutes=30).execute(request)
        elif algo == "limit_first":
            result = await LimitFirstExecution(self.broker, offset_bps=5, fallback_seconds=30).execute(request)
        elif algo == "rl_exec" and _RL_EXEC_AVAILABLE:
            fills = await RLExecution(self.broker, agent=get_rl_agent()).execute(request, signal_price)
            # Aggregate fills into a single OrderResult for compatibility
            if fills:
                total_qty = sum(f["qty"] for f in fills)
                avg_price = sum(f["qty"] * f["price"] for f in fills) / max(total_qty, 1e-9)
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

        if self.slippage_tracker:
            await self.slippage_tracker.record_fill(request, result)

        return result

    def _select_algorithm(self, request: OrderRequest) -> str:
        """Determine the most suitable execution algorithm for a given request.

        The selection logic follows a hierarchy:
        * Explicit `execution_algo` overrides any automatic choice.
        * Very large orders (>= $100k) prefer RL execution when available,
          otherwise fall back to TWAP.
        * Mid‑size orders (>= $5k and < $100k) use Almgren‑Chriss.
        * Limit orders with a price use the limit‑first strategy.
        * All other cases default to a market order.

        Parameters
        ----------
        request : OrderRequest
            The order for which an algorithm is to be selected.

        Returns
        -------
        str
            The identifier of the chosen algorithm (e.g., ``"twap"``, ``"almgren_chriss"``,
            ``"limit_first"``, ``"rl_exec"``, or ``"market"``).
        """
        # Use signal_price if available (set on OrderRequest.metadata), then limit_price,
        # then stop_price, then fall back to $50 (mid‑range ETF proxy, less wrong than $100)
        ref_price = (
            request.limit_price
            or request.stop_price
            or (asdict(request).get("metadata") or {}).get("signal_price")
            or 50.0
        )
        estimated_usd = request.quantity * ref_price

        if request.execution_algo and request.execution_algo not in ("auto", ""):
            return request.execution_algo   # explicit user/strategy override
        elif estimated_usd >= 100_000 and _RL_EXEC_AVAILABLE:
            return "rl_exec"   # RL agent for very large orders (better than TWAP)
        elif estimated_usd >= 100_000:
            return "twap"
        elif 5_000 <= estimated_usd < 100_000:
            return "almgren_chriss"   # optimal IS minimisation for mid-size orders
        elif request.order_type == "limit" and request.limit_price:
            return "limit_first"
        else:
            return "market"

    async def _execute_almgren_chriss(self, request: OrderRequest) -> OrderResult:
        """Execute an order using the Almgren‑Chriss optimal execution trajectory.

        The algorithm splits the total quantity into a series of slices that are
        submitted as market orders. Each slice respects the optimal schedule
        computed by the Almgren‑Chriss model.

        Parameters
        ----------
        request : OrderRequest
            The original order to be executed.

        Returns
        -------
        OrderResult
            Aggregated result of the sliced execution, including total filled
            quantity and average fill price.
        """
        import asyncio

        # Estimate sigma from metadata if available, default 2%
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
            # Use market slices — adding "limit" without a price causes broker rejection.
            # AC's alpha comes from the optimal schedule, not from limit orders.
            slice_req = OrderRequest(
                **{**asdict(request), "quantity": float(slice_qty), "order_type": "market", "limit_price": None}
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
            except Exception as e:
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
            status="filled" if total_filled >= request.quantity * 0.95 else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )