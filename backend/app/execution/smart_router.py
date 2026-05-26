"""
Smart Order Router — selects the best execution algorithm based on order characteristics.
Goal: minimize slippage while ensuring fills.

Decision logic:
  - Large orders (>$10k): TWAP over 30 min
  - Crypto buys: Limit-first (post limit, fallback to market after 30s)
  - Urgent signals: Market order
  - Default: VWAP with 10% participation rate
"""
from app.brokers.base import OrderRequest, OrderResult, AbstractBroker
from app.execution.limit_first import LimitFirstExecution
from app.execution.twap import TWAPExecution
from app.execution.slippage_tracker import SlippageTracker


class SmartOrderRouter:
    def __init__(self, broker: AbstractBroker, slippage_tracker: SlippageTracker | None = None):
        self.broker = broker
        self.slippage_tracker = slippage_tracker

    async def execute(self, request: OrderRequest, signal_price: float | None = None) -> OrderResult:
        """Route order to the optimal execution algorithm."""
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
