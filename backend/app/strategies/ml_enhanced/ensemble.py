"""Ensemble strategy: pure ML signal from all models combined."""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.ml.inference import get_inference_service


class EnsembleStrategy(AbstractStrategy):
    name = "ensemble"
    display_name = "Ensemble ML (LSTM + XGB + Lorentzian)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0
    confidence_threshold = 0.70  # higher bar for pure ML

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result is None or ml_result["prediction"] == "neutral":
                return None
            if ml_result["confidence"] < self.confidence_threshold:
                return None
            return Signal(
                symbol=symbol,
                side="buy" if ml_result["prediction"] == "up" else "sell",
                confidence=ml_result["confidence"],
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata=ml_result,
            )
        except Exception:
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        empty = pd.Series(False, index=df.index)
        return BacktestSignals(entries=empty, exits=empty)
