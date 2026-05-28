"""
Continuous strategy runner: one asyncio task per (strategy, symbol) pair.
Scales to hundreds of concurrent strategy+symbol combinations.
"""
from __future__ import annotations
import asyncio
from app.strategies import STRATEGY_REGISTRY
from app.redis_client import get_redis
from app.ws.manager import manager
from app.utils.logging import logger


class ContinuousStrategyRunner:
    def __init__(self, broker, risk_manager=None):
        self.broker = broker
        self.risk_manager = risk_manager
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self, active_strategies: list[dict]) -> None:
        """
        active_strategies: list of {name, symbols, params, tick_interval_seconds, confidence_threshold}
        """
        self._running = True
        self._tasks = [
            asyncio.create_task(
                self._run_loop(s["name"], symbol, s.get("params", {}),
                               s.get("tick_interval_seconds", 60),
                               s.get("confidence_threshold", 0.6))
            )
            for s in active_strategies
            for symbol in s.get("symbols", [])
        ]
        logger.info("Strategy runner started", tasks=len(self._tasks))
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()

    async def _run_loop(self, strategy_name: str, symbol: str, params: dict,
                        tick_interval: int, confidence_threshold: float) -> None:
        cache = await get_redis()
        strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
        if not strategy_cls:
            logger.error("Unknown strategy", name=strategy_name)
            return

        strategy = strategy_cls(**params) if params else strategy_cls()
        logger.info("Strategy loop started", strategy=strategy_name, symbol=symbol)

        while self._running:
            try:
                # Get OHLCV from Redis cache or broker
                import pandas as pd
                ohlcv = await cache.get_ohlcv(symbol)
                if ohlcv:
                    df = pd.DataFrame(ohlcv)
                else:
                    from datetime import datetime, timedelta, timezone
                    end = datetime.now(timezone.utc)
                    start = end - timedelta(days=90)
                    bars = await self.broker.get_historical(symbol, "1h", start, end)
                    df = pd.DataFrame(bars)

                if df is None or len(df) < 30:
                    await asyncio.sleep(tick_interval)
                    continue

                signal = await strategy.analyze(df, symbol)
                if signal and signal.confidence >= confidence_threshold:
                    alert = {
                        "type": "signal",
                        "strategy": strategy_name,
                        "symbol": symbol,
                        "side": signal.side,
                        "confidence": round(signal.confidence, 4),
                        "target_price": signal.target_price,
                        "stop_loss": signal.stop_loss,
                    }
                    await manager.broadcast("alerts", alert)
                    logger.info("Signal generated", **alert)

                    from app.notifications.slack import slack
                    from app.notifications.tracker import tracker
                    tracker.record("signal_fired", "signal",
                                    f"{strategy_name} → {symbol} {signal.side} (conf={signal.confidence:.2f})")
                    await slack.notify_signal(strategy_name, symbol, signal.side,
                                                signal.confidence, signal.target_price)

                    # ── Submit order through risk-gated smart router ──────────
                    if self.broker is not None:
                        from app.brokers.base import OrderRequest
                        from app.execution.smart_router import SmartOrderRouter
                        router = SmartOrderRouter(
                            broker=self.broker,
                            risk_manager=self.risk_manager,
                        )
                        order_req = OrderRequest(
                            symbol=symbol,
                            quantity=signal.quantity or 1,
                            side=signal.side,
                            order_type=signal.order_type or "market",
                            limit_price=signal.target_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            risk_bucket=strategy.risk_bucket,
                        )
                        result = await router.execute(order_req, signal_price=signal.target_price)
                        if result:
                            logger.info("Order submitted", strategy=strategy_name, symbol=symbol,
                                        order_id=getattr(result, "order_id", "?"),
                                        side=signal.side, qty=order_req.quantity)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Strategy loop error", strategy=strategy_name, symbol=symbol, error=str(e))

            await asyncio.sleep(tick_interval)
