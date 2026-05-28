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
        else:
            result = await self.broker.place_order(request)

        if self.slippage_tracker:
            await self.slippage_tracker.record_fill(request, result)

        return result

    def _select_algorithm(self, request: OrderRequest) -> str:
        estimated_usd = request.quantity * (request.limit_price or 100)

        if estimated_usd > 10_000:
            return "twap"
        elif request.execution_algo and request.execution_algo != "auto":
            return request.execution_algo   # user override
        elif request.order_type == "limit" and request.limit_price:
            return "limit_first"
        else:
            return "market"
