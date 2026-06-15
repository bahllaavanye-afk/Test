"""ML-filtered breakout strategy."""
import pandas as pd

from app.ml.inference import get_inference_service
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.strategies.manual.breakout import BreakoutStrategy


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
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result and ml_result["confidence"] > 0.65 and ml_result["prediction"] == "up":
                base_signal.confidence = min(0.92, (base_signal.confidence + ml_result["confidence"]) / 2)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                return base_signal
        except Exception:
            return base_signal
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        return self._base.backtest_signals(df)
