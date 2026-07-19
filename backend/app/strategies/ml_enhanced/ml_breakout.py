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
        signal_generated = False
        confidence = None

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
                    signal_generated = True
                    confidence = base_signal.confidence
                else:
                    # No ML enhancement applied; keep original signal
                    signal_generated = True
                    confidence = base_signal.confidence
            except Exception as exc:
                logger.exception(
                    "ML inference failed for %s; falling back to base signal", symbol
                )
                signal_generated = True
                confidence = base_signal.confidence
                return base_signal
            return base_signal
        else:
            # No base signal produced
            signal_generated = False

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "MLBreakout analyze completed",
            extra={
                "symbol": symbol,
                "duration_ms": round(duration_ms, 2),
                "signal_generated": signal_generated,
                "confidence": confidence,
            },
        )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        start_time = time.perf_counter()
        result = self._base.backtest_signals(df)

        # Attempt to extract useful metrics safely
        signal_count = getattr(result, "signal_count", None)
        if signal_count is None:
            # Fallback: try to infer length if result is iterable
            try:
                signal_count = len(list(result))
            except Exception:
                signal_count = None

        pnl = getattr(result, "pnl", None)
        if pnl is None:
            pnl = getattr(result, "profit_loss", None)

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "MLBreakout backtest completed",
            extra={
                "duration_ms": round(duration_ms, 2),
                "signal_count": signal_count,
                "pnl": pnl,
            },
        )
        return result