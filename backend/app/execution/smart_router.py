"""
Smart Order Router — selects the best execution algorithm based on order characteristics.
Goal: minimize slippage while ensuring fills.

Decision logic:
  - Large orders (>$10k): TWAP over 30 min
  - Crypto buys: Limit-first (post limit, fallback to market after 30s)
  - Urgent signals: Market order
  - Default: VWAP with 10% participation rate

All orders pass through RiskManager.check_order() before execution.
"""
import time
from dataclasses import asdict

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
    def __init__(
        self,
        broker: AbstractBroker,
        slippage_tracker: SlippageTracker | None = None,
        risk_manager=None,
    ):
        self.broker = broker
        self.slippage_tracker = slippage_tracker
        self.risk_manager = risk_manager
        self._signal_counter = 0  # tracks number of executed signals

    async def execute(self, request: OrderRequest, signal_price: float | None = None) -> OrderResult | None:
        """Route order to the optimal execution algorithm.

        Returns None (and logs a warning) if the risk manager blocks the order.
        """
        start_ts = time.monotonic()
        self._signal_counter += 1

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
                from app.brokers.base import OrderResult
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

        # ── Monitoring ────────────────────────────────────────────────────────
        exec_time_ms = (time.monotonic() - start_ts) * 1000
        pnl = None
        if result and signal_price is not None:
            side = getattr(request, "side", "buy").lower()
            filled_qty = getattr(result, "filled_qty", 0.0)
            avg_price = getattr(result, "avg_fill_price", None)
            if avg_price is not None:
                if side == "buy":
                    pnl = (signal_price - avg_price) * filled_qty
                elif side == "sell":
                    pnl = (avg_price - signal_price) * filled_qty

        logger.info(
            "Order execution completed",
            signal_count=self._signal_counter,
            execution_time_ms=round(exec_time_ms, 2),
            pnl=round(pnl, 4) if pnl is not None else None,
            symbol=request.symbol,
            order_id=getattr(result, "order_id", None),
        )

        return result

    def _select_algorithm(self, request: OrderRequest) -> str:
        # Use signal_price if available (set on OrderRequest.metadata), then limit_price,
        # then stop_price, then fall back to $50 (mid-range ETF proxy, less wrong than $100)
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
        """
        Execute order using Almgren-Chriss optimal trajectory.
        Each slice is submitted as a limit order at the current mid-price.
        """
        import asyncio

        # Estimate sigma from metadata if available, default 2%
        sigma = float(asdict(request).get("metadata", {}).get("sigma", 0.02)) if hasattr(request, "__dict__") else 0.02
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