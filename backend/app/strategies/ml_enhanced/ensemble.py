"""Ensemble strategy: pure ML signal from all models combined."""
import logging
import time

import pandas as pd

from app.ml.inference import get_inference_service
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

logger = logging.getLogger(__name__)


class EnsembleStrategy(AbstractStrategy):
    name = "ensemble"
    display_name = "Ensemble ML (LSTM + XGB + Lorentzian)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0
    confidence_threshold = 0.70  # higher bar for pure ML

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signal_count = 0

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        start_time = time.time()
        signal: Signal | None = None
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result is None or ml_result["prediction"] == "neutral":
                return None
            if ml_result["confidence"] < self.confidence_threshold:
                return None
            signal = Signal(
                symbol=symbol,
                side="buy" if ml_result["prediction"] == "up" else "sell",
                confidence=ml_result["confidence"],
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata=ml_result,
            )
            self._signal_count += 1
            return signal
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "EnsembleStrategy analyze completed",
                extra={
                    "symbol": symbol,
                    "signal_generated": signal is not None,
                    "signal_count": self._signal_count,
                    "execution_time_ms": round(elapsed_ms, 2),
                },
            )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        empty = pd.Series(False, index=df.index)
        backtest = BacktestSignals(entries=empty, exits=empty)
        logger.info(
            "EnsembleStrategy backtest_signals executed",
            extra={"data_points": len(df), "entries": 0, "exits": 0},
        )
        return backtest