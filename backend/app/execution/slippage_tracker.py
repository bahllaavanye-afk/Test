"""Tracks realized slippage vs expected fill price per order."""
import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from app.brokers.base import OrderRequest, OrderResult
from app.models.slippage import SlippageRecord
from app.utils.logging import logger


class SlippageTracker:
    def __init__(self, db: AsyncSession | None = None):
        self.db = db
        self._signal_prices: dict[str, float] = {}

    async def record_signal_price(self, request: OrderRequest, signal_price: float) -> None:
        key = f"{request.account_id}:{request.symbol}"
        self._signal_prices[key] = signal_price

    async def record_fill(self, request: OrderRequest, result: OrderResult) -> None:
        if not result.avg_fill_price:
            return
        key = f"{request.account_id}:{request.symbol}"
        signal_price = self._signal_prices.pop(key, None)

        if signal_price and result.avg_fill_price:
            if request.side == "buy":
                slippage_bps = (result.avg_fill_price - signal_price) / signal_price * 10000
            else:
                slippage_bps = (signal_price - result.avg_fill_price) / signal_price * 10000

            logger.info("Slippage recorded",
                        symbol=request.symbol,
                        expected=signal_price,
                        fill=result.avg_fill_price,
                        slippage_bps=round(slippage_bps, 2),
                        algo=request.execution_algo)

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
                    created_at=datetime.now(timezone.utc),
                )
                self.db.add(record)
                await self.db.commit()
