"""ML-filtered breakout strategy."""
import logging
import time
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.breakout import BreakoutStrategy
from app.ml.inference import get_inference_service

logger = logging.getLogger(__name__)


class MLBreakoutStrategy(AbstractStrategy):
    name = "ml_breakout"
    display_name = "ML Breakout (Volume + Ensemble)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._base = BreakoutStrategy(params)
        self._signal_count = 0

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        start_time = time.perf_counter()
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            elapsed = time.perf_counter() - start_time
            logger.info(
                "MLBreakoutStrategy analyze - no base signal",
                extra={"symbol": symbol, "execution_time_s": elapsed, "signal_count": self._signal_count},
            )
            return None
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result and ml_result["confidence"] > 0.65 and ml_result["prediction"] == "up":
                base_signal.confidence = min(
                    0.92, (base_signal.confidence + ml_result["confidence"]) / 2
                )
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
        except Exception:
            # Preserve existing behavior: fallback to base signal on any error
            pass

        # Log metrics if a signal is produced
        if base_signal:
            self._signal_count += 1
            elapsed = time.perf_counter() - start_time
            pnl = getattr(base_signal, "pnl", None)
            logger.info(
                "MLBreakoutStrategy analyze - signal generated",
                extra={
                    "symbol": symbol,
                    "execution_time_s": elapsed,
                    "signal_count": self._signal_count,
                    "pnl": pnl,
                },
            )
            return base_signal
        else:
            elapsed = time.perf_counter() - start_time
            logger.info(
                "MLBreakoutStrategy analyze - no ML signal",
                extra={"symbol": symbol, "execution_time_s": elapsed, "signal_count": self._signal_count},
            )
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        return self._base.backtest_signals(df)