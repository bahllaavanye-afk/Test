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
from app.brokers.base import OrderRequest, OrderResult, AbstractBroker
from app.execution.limit_first import LimitFirstExecution
from app.execution.twap import TWAPExecution
from app.execution.slippage_tracker import SlippageTracker
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

    async def execute(self, request: OrderRequest, signal_price: float | None = None) -> OrderResult | None:
        """Route order to the optimal execution algorithm.

        Returns None (and logs a warning) if the risk manager blocks the order.
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

        if algo == "twap":
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

        return result

    def _select_algorithm(self, request: OrderRequest) -> str:
        estimated_usd = request.quantity * (request.limit_price or 100)

        if request.execution_algo and request.execution_algo not in ("auto", ""):
            return request.execution_algo   # explicit user/strategy override
        elif estimated_usd > 10_000 and _RL_EXEC_AVAILABLE:
            return "rl_exec"   # RL agent for large orders (better than TWAP)
        elif estimated_usd > 10_000:
            return "twap"
        elif request.order_type == "limit" and request.limit_price:
            return "limit_first"
        else:
            return "market"
