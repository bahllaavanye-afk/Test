"""ML-enhanced momentum: Jegadeesh-Titman signals filtered by LSTM + XGBoost ensemble."""
import pandas as pd

from app.ml.inference import get_inference_service
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.strategies.manual.momentum import MomentumStrategy


class MLMomentumStrategy(AbstractStrategy):
    name = "ml_momentum"
    display_name = "ML Momentum (LSTM + XGBoost Filter)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.65

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._base = MomentumStrategy(params)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        # Apply ML filter
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result is None or ml_result["prediction"] == "neutral":
                return None
            # Only pass if ML agrees with indicator direction
            if ml_result["prediction"] == "up" and base_signal.side == "buy":
                base_signal.confidence = min(0.95, (base_signal.confidence + ml_result["confidence"]) / 2)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                base_signal.metadata["ml_confidence"] = ml_result["confidence"]
                return base_signal
            elif ml_result["prediction"] == "down" and base_signal.side == "sell":
                base_signal.confidence = min(0.95, (base_signal.confidence + ml_result["confidence"]) / 2)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                base_signal.metadata["ml_confidence"] = ml_result["confidence"]
                return base_signal
        except Exception:
            pass  # Fall back to base signal if ML unavailable
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        # For backtesting without live ML: use base signals as proxy
        return self._base.backtest_signals(df)
