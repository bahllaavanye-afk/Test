"""ML-filtered breakout strategy."""
import pandas as pd
from typing import Optional

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

    async def analyze(self, data: Optional[pd.DataFrame], symbol: Optional[str]) -> Signal | None:
        # Guard against None or empty inputs
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return None
        if not symbol:
            return None

        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None

        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)

            # Validate ml_result structure and values
            if (
                isinstance(ml_result, dict)
                and isinstance(ml_result.get("confidence"), (int, float))
                and ml_result.get("prediction") is not None
            ):
                confidence = ml_result["confidence"]
                prediction = ml_result["prediction"]

                if confidence > 0.65 and prediction == "up":
                    # Combine confidences safely, capping at 0.92
                    combined_conf = (base_signal.confidence + confidence) / 2
                    base_signal.confidence = min(0.92, combined_conf)
                    base_signal.strategy_name = self.name
                    base_signal.strategy_type = self.strategy_type
                    return base_signal
        except Exception:
            # On any failure, fall back to the base signal
            return base_signal

        # If ML conditions not met, do not emit a signal
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        return self._base.backtest_signals(df)