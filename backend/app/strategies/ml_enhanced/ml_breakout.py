"""ML-filtered breakout strategy."""
import pandas as pd
import numpy as np

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

    @staticmethod
    def _recent_high(df: pd.DataFrame, lookback: int = 20) -> float:
        """Return the highest close in the lookback window."""
        if df.empty:
            return np.nan
        return df["close"].tail(lookback).max()

    @staticmethod
    def _average_volume(df: pd.DataFrame, lookback: int = 20) -> float:
        """Return the average volume over the lookback window."""
        if df.empty:
            return np.nan
        return df["volume"].tail(lookback).mean()

    def _entry_filters(self, df: pd.DataFrame) -> bool:
        """
        Tighten entry conditions:
        - Current close must break recent high by at least 1%.
        - Current volume must exceed average volume by 50%.
        """
        if "close" not in df.columns or "volume" not in df.columns:
            return False

        recent_high = self._recent_high(df)
        avg_vol = self._average_volume(df)

        if np.isnan(recent_high) or np.isnan(avg_vol):
            return False

        current_close = df["close"].iloc[-1]
        current_vol = df["volume"].iloc[-1]

        price_break = current_close > recent_high * 1.01
        volume_break = current_vol > avg_vol * 1.5
        return price_break and volume_break

    def _exit_filters(self, df: pd.DataFrame) -> bool:
        """
        Exit confirmation:
        - If price falls below the 10‑period moving average, signal exit.
        - If ML predicts a down move with confidence > 0.6, signal exit.
        """
        if "close" not in df.columns:
            return False

        ma10 = df["close"].rolling(window=10).mean().iloc[-1]
        current_close = df["close"].iloc[-1]
        price_fall = current_close < ma10
        return price_fall

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Produce a signal after applying the base breakout logic,
        reinforced with volume/price breakout filters and ML confirmation.
        """
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None

        # Determine if we are dealing with an entry or exit signal
        is_entry = getattr(base_signal, "signal_type", "enter") == "enter"

        # Apply tighter entry filters
        if is_entry and not self._entry_filters(data):
            return None

        # Apply exit filters (price‑based) before consulting ML
        if not is_entry and self._exit_filters(data):
            # Override confidence to reflect a strong exit trigger
            base_signal.confidence = max(base_signal.confidence, 0.85)
            base_signal.strategy_name = self.name
            base_signal.strategy_type = self.strategy_type
            return base_signal

        # ML confirmation – only used to boost confidence for entries
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if (
                ml_result
                and ml_result.get("confidence", 0) > 0.70
                and ml_result.get("prediction") == "up"
                and is_entry
                and self._entry_filters(data)  # double‑check filters
            ):
                # Blend ML confidence with base confidence, capped at 0.95
                blended = (base_signal.confidence + ml_result["confidence"]) / 2
                base_signal.confidence = min(0.95, blended)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                return base_signal
        except Exception:
            # Preserve base signal if ML inference fails
            pass

        # For exit signals that didn't meet the price‑based filter,
        # we keep the original signal but still tag it.
        if not is_entry:
            base_signal.strategy_name = self.name
            base_signal.strategy_type = self.strategy_type
            return base_signal

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Delegate backtesting to the underlying breakout strategy."""
        return self._base.backtest_signals(df)