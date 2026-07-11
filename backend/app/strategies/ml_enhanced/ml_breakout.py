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

    async def analyze(self, data: pd.DataFrame | None, symbol: str | None) -> Signal | None:
        # Guard against None or empty inputs
        if data is None or data.empty:
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
                and ml_result.get("confidence") is not None
                and ml_result.get("prediction") is not None
                and ml_result["confidence"] > 0.65
                and ml_result["prediction"] == "up"
            ):
                # Combine confidences, ensuring we stay within bounds
                combined_conf = (base_signal.confidence + ml_result["confidence"]) / 2
                base_signal.confidence = min(0.92, combined_conf)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                return base_signal
        except Exception:
            # In case of any failure, fall back to the base signal
            return base_signal

        return None

    def backtest_signals(self, df: pd.DataFrame | None) -> BacktestSignals:
        # Guard against None or empty DataFrames in backtesting
        if df is None or df.empty:
            return BacktestSignals([], [])
        return self._base.backtest_signals(df)