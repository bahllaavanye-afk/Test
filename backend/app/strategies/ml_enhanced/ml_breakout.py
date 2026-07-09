"""ML-filtered breakout strategy."""
import logging
import time
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.breakout import BreakoutStrategy
from app.ml.inference import get_inference_service

_logger = logging.getLogger(__name__)


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
            _logger.info(
                "MLBreakout analyze - no base signal",
                extra={
                    "symbol": symbol,
                    "duration_ms": (time.perf_counter() - start_time) * 1000,
                    "signal_generated": False,
                },
            )
            return None

        final_signal: Signal | None = None
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if (
                ml_result
                and ml_result["confidence"] > 0.65
                and ml_result["prediction"] == "up"
            ):
                base_signal.confidence = min(
                    0.92, (base_signal.confidence + ml_result["confidence"]) / 2
                )
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                final_signal = base_signal
        except Exception:
            # Preserve base signal on any inference error
            final_signal = base_signal

        duration_ms = (time.perf_counter() - start_time) * 1000
        _logger.info(
            "MLBreakout analyze completed",
            extra={
                "symbol": symbol,
                "duration_ms": duration_ms,
                "signal_generated": final_signal is not None,
                "signal_confidence": getattr(final_signal, "confidence", None),
            },
        )
        return final_signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        start_time = time.perf_counter()
        backtest_result = self._base.backtest_signals(df)

        # Attempt to extract signal count; fallback to None if attribute not present
        signal_count = getattr(backtest_result, "signals", None)
        if isinstance(signal_count, (list, tuple)):
            signal_count = len(signal_count)

        duration_ms = (time.perf_counter() - start_time) * 1000
        _logger.info(
            "MLBreakout backtest_signals completed",
            extra={
                "duration_ms": duration_ms,
                "signal_count": signal_count,
                # Assuming BacktestSignals may expose P&L; log if available
                "pnl": getattr(backtest_result, "pnl", None),
            },
        )
        return backtest_result