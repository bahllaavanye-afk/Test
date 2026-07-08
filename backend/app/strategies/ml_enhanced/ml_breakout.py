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
        start_time = time.time()
        signal: Signal | None = None

        base_signal = await self._base.analyze(data, symbol)
        if base_signal:
            try:
                inference = get_inference_service()
                ml_result = await inference.predict(data, symbol)
                if (
                    ml_result
                    and ml_result["confidence"] > 0.65
                    and ml_result["prediction"] == "up"
                ):
                    base_signal.confidence = min(
                        0.92,
                        (base_signal.confidence + ml_result["confidence"]) / 2,
                    )
                    base_signal.strategy_name = self.name
                    base_signal.strategy_type = self.strategy_type
                    signal = base_signal
                else:
                    signal = base_signal
            except Exception:
                signal = base_signal

        duration_ms = (time.time() - start_time) * 1000
        signal_count = 1 if signal else 0
        pnl = getattr(signal, "pnl", None) if signal else None

        logger.info(
            "MLBreakout analyze completed",
            extra={"signal_count": signal_count, "duration_ms": duration_ms, "pnl": pnl},
        )
        return signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        result = self._base.backtest_signals(df)
        try:
            signal_count = len(result.signals) if hasattr(result, "signals") else 0
        except Exception:
            signal_count = 0
        logger.info(
            "MLBreakout backtest signals generated",
            extra={"signal_count": signal_count},
        )
        return result