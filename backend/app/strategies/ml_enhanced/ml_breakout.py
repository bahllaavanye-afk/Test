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

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        start_time = time.perf_counter()
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "MLBreakout - no base signal",
                extra={
                    "symbol": symbol,
                    "signal_count": 0,
                    "execution_time_ms": duration_ms,
                },
            )
            return None
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result and ml_result["confidence"] > 0.65 and ml_result["prediction"] == "up":
                base_signal.confidence = min(
                    0.92,
                    (base_signal.confidence + ml_result["confidence"]) / 2,
                )
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "MLBreakout - signal generated",
                    extra={
                        "symbol": symbol,
                        "signal_count": 1,
                        "execution_time_ms": duration_ms,
                        "pnl": getattr(base_signal, "pnl", None),
                    },
                )
                return base_signal
        except Exception:
            logger.exception("MLBreakout - inference error")
            # Fallback to base signal without ML adjustment
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "MLBreakout - base signal returned",
            extra={
                "symbol": symbol,
                "signal_count": 1,
                "execution_time_ms": duration_ms,
                "pnl": getattr(base_signal, "pnl", None),
            },
        )
        return base_signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        signals = self._base.backtest_signals(df)
        signal_count = len(signals) if hasattr(signals, "__len__") else 0
        logger.info(
            "MLBreakout - backtest signals",
            extra={"signal_count": signal_count},
        )
        return signals