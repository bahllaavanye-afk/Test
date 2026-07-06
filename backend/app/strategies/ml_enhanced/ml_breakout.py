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

    @staticmethod
    def _price_breakout(data: pd.DataFrame, pct: float = 0.01) -> bool:
        """Return True if the latest close breaks above the previous high by a given percent."""
        if data.shape[0] < 2:
            return False
        latest_close = data["close"].iloc[-1]
        prev_high = data["high"].iloc[-2]
        return latest_close > prev_high * (1 + pct)

    @staticmethod
    def _volume_surge(data: pd.DataFrame, multiplier: float = 1.5) -> bool:
        """Return True if the latest volume exceeds the rolling average by a multiplier."""
        if "volume" not in data.columns:
            return False
        recent_vol = data["volume"].iloc[-1]
        avg_vol = data["volume"].rolling(window=20, min_periods=1).mean().iloc[-1]
        return recent_vol > avg_vol * multiplier

    @staticmethod
    def _price_below_sma(data: pd.DataFrame, window: int = 20) -> bool:
        """Return True if the latest close is below its simple moving average."""
        if data.shape[0] < window:
            return False
        sma = data["close"].rolling(window=window, min_periods=1).mean().iloc[-1]
        latest_close = data["close"].iloc[-1]
        return latest_close < sma

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Generate a signal if breakout conditions and ML confirmation are satisfied."""
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None

        # Basic technical filters to tighten entry criteria
        if base_signal.confidence < 0.6:
            return None
        if not self._price_breakout(data):
            return None
        if not self._volume_surge(data):
            return None

        # Apply ML confirmation if available
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if (
                ml_result
                and ml_result.get("prediction") == "up"
                and ml_result.get("confidence", 0) > 0.70
            ):
                # Blend confidences, but cap to avoid over‑confidence
                blended = (base_signal.confidence + ml_result["confidence"]) / 2
                base_signal.confidence = min(0.92, blended)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
        except Exception:
            # If ML inference fails, retain the base signal unchanged
            pass

        # Exit‑related sanity check: reduce confidence if price falls below SMA
        if self._price_below_sma(data):
            base_signal.confidence *= 0.8

        # Discard signal if confidence falls below a safe threshold after adjustments
        if base_signal.confidence < 0.55:
            return None

        return base_signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Delegate backtesting to the underlying breakout strategy."""
        return self._base.backtest_signals(df)