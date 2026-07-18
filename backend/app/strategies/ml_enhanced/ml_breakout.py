"""ML-filtered breakout strategy."""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.breakout import BreakoutStrategy
from app.ml.inference import get_inference_service

# Constants
NAME = "ml_breakout"
DISPLAY_NAME = "ML Breakout (Volume + Ensemble)"
MARKET_TYPE = "equity"
STRATEGY_TYPE = "ml_enhanced"
RISK_BUCKET = "directional"
TICK_INTERVAL_SECONDS = 900.0
CONFIDENCE_THRESHOLD = 0.65
PREDICTION_UP = "up"
CONFIDENCE_CAP = 0.92


class MLBreakoutStrategy(AbstractStrategy):
    name = NAME
    display_name = DISPLAY_NAME
    market_type = MARKET_TYPE
    strategy_type = STRATEGY_TYPE
    risk_bucket = RISK_BUCKET
    tick_interval_seconds = TICK_INTERVAL_SECONDS

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._base = BreakoutStrategy(params)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if (
                ml_result
                and ml_result["confidence"] > CONFIDENCE_THRESHOLD
                and ml_result["prediction"] == PREDICTION_UP
            ):
                base_signal.confidence = min(
                    CONFIDENCE_CAP,
                    (base_signal.confidence + ml_result["confidence"]) / 2,
                )
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                return base_signal
        except Exception:
            return base_signal
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        return self._base.backtest_signals(df)