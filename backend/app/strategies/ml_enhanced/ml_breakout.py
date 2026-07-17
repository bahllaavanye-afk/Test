"""ML-filtered breakout strategy."""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.breakout import BreakoutStrategy
from app.ml.inference import get_inference_service


class MLBreakoutStrategy(AbstractStrategy):
    """Combines a traditional breakout strategy with ML predictions."""

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
        """Generate a signal, augmenting it with ML confidence when applicable."""
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None

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
                return base_signal
        except Exception:
            # Preserve the base signal if ML inference fails.
            return base_signal

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Delegate backtesting to the underlying breakout strategy."""
        return self._base.backtest_signals(df)