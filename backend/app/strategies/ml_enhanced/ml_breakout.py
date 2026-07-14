"""ML-filtered breakout strategy."""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.breakout import BreakoutStrategy
from app.ml.inference import get_inference_service


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
        # Guard against None or empty inputs
        if data is None or data.empty or not symbol:
            return None

        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None

        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)

            # Ensure ml_result contains expected keys
            confidence = ml_result.get("confidence") if isinstance(ml_result, dict) else None
            prediction = ml_result.get("prediction") if isinstance(ml_result, dict) else None

            if confidence is not None and prediction == "up" and confidence > 0.65:
                # Combine confidences while respecting upper bound
                combined_conf = (base_signal.confidence + confidence) / 2
                base_signal.confidence = min(0.92, combined_conf)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                return base_signal
        except Exception:
            # Fallback to base signal if any error occurs
            return base_signal

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        # Guard against None or empty dataframe
        if df is None or df.empty:
            return BacktestSignals([])
        return self._base.backtest_signals(df)