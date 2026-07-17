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
        start_time = time.time()
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            duration = time.time() - start_time
            logger.info(
                "MLBreakoutStrategy analyze - no base signal",
                extra={"symbol": symbol, "duration_s": duration, "signal_count": self._signal_count},
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
                self._signal_count += 1
                duration = time.time() - start_time
                logger.info(
                    "MLBreakoutStrategy signal generated",
                    extra={
                        "symbol": symbol,
                        "duration_s": duration,
                        "signal_count": self._signal_count,
                        "ml_confidence": ml_result["confidence"],
                        "base_confidence": base_signal.confidence,
                    },
                )
                return base_signal
        except Exception as e:
            duration = time.time() - start_time
            logger.info(
                "MLBreakoutStrategy analysis exception, returning base signal",
                extra={
                    "symbol": symbol,
                    "duration_s": duration,
                    "signal_count": self._signal_count,
                    "error": str(e),
                },
            )
            return base_signal
        duration = time.time() - start_time
        logger.info(
            "MLBreakoutStrategy analyze - no ML signal",
            extra={"symbol": symbol, "duration_s": duration, "signal_count": self._signal_count},
        )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        signals = self._base.backtest_signals(df)
        logger.info(
            "MLBreakoutStrategy backtest completed",
            extra={"signal_count": len(signals) if hasattr(signals, '__len__') else 'N/A'},
        )
        return signals