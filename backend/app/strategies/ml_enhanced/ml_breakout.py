"""ML-filtered breakout strategy."""
import pandas as pd
from pandas.util import hash_pandas_object
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
        # Simple in‑memory cache: symbol -> (data_hash, ml_result)
        self._ml_cache: dict[str, tuple[int, dict]] = {}

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None

        # If the base signal already has very high confidence, skip ML inference.
        if base_signal.confidence >= 0.9:
            return base_signal

        # Compute a lightweight hash of the incoming DataFrame for caching.
        try:
            data_hash = int(hash_pandas_object(data, index=True).sum())
        except Exception:
            data_hash = 0

        cached = self._ml_cache.get(symbol)
        if cached and cached[0] == data_hash:
            ml_result = cached[1]
        else:
            try:
                inference = get_inference_service()
                ml_result = await inference.predict(data, symbol)
                # Store result in cache even if None to avoid repeated calls on failure.
                self._ml_cache[symbol] = (data_hash, ml_result)
            except Exception:
                ml_result = None

        if ml_result and ml_result.get("confidence", 0) > 0.65 and ml_result.get("prediction") == "up":
            # Blend confidences, capping at 0.92 as before.
            blended_conf = min(0.92, (base_signal.confidence + ml_result["confidence"]) / 2)
            base_signal.confidence = blended_conf
            base_signal.strategy_name = self.name
            base_signal.strategy_type = self.strategy_type
            return base_signal

        # If ML inference fails or does not meet criteria, fall back to the base signal.
        return base_signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        return self._base.backtest_signals(df)