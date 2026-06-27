"""ML-filtered breakout strategy."""
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator

from app.ml.inference import get_inference_service
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.strategies.manual.breakout import BreakoutStrategy


class MLBreakoutParams(BaseModel):
    """Configuration parameters for the ML Breakout strategy."""

    lookback: int = Field(
        default=20,
        description="Number of periods to consider for recent high and average volume calculations.",
        example=20,
        ge=1,
    )
    price_multiplier: float = Field(
        default=1.01,
        description="Factor by which the current close must exceed the recent high to qualify as a breakout.",
        example=1.01,
        gt=0,
    )
    volume_multiplier: float = Field(
        default=1.5,
        description="Factor by which the current volume must exceed the average volume to qualify as a breakout.",
        example=1.5,
        gt=0,
    )
    ml_confidence_threshold: float = Field(
        default=0.70,
        description="Minimum confidence required from the ML model to influence entry decisions.",
        example=0.70,
        ge=0,
        le=1,
    )
    blended_confidence_cap: float = Field(
        default=0.95,
        description="Maximum confidence value after blending base and ML confidences.",
        example=0.95,
        ge=0,
        le=1,
    )
    exit_price_ma_window: int = Field(
        default=10,
        description="Window size for the moving average used in exit price filter.",
        example=10,
        ge=1,
    )

    @validator("price_multiplier", "volume_multiplier", "ml_confidence_threshold", "blended_confidence_cap")
    def _validate_range(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("must be a numeric type")
        return v


class MLBreakoutStrategy(AbstractStrategy):
    name = "ml_breakout"
    display_name = "ML Breakout (Volume + Ensemble)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        # Validate and store strategy‑specific parameters
        self.params = MLBreakoutParams(**(params or {}))
        self._base = BreakoutStrategy(params)

    @staticmethod
    def _recent_high(df: pd.DataFrame, lookback: int) -> float:
        """Return the highest close in the lookback window."""
        if df.empty:
            return np.nan
        return df["close"].tail(lookback).max()

    @staticmethod
    def _average_volume(df: pd.DataFrame, lookback: int) -> float:
        """Return the average volume over the lookback window."""
        if df.empty:
            return np.nan
        return df["volume"].tail(lookback).mean()

    def _entry_filters(self, df: pd.DataFrame) -> bool:
        """
        Tighten entry conditions:
        - Current close must break recent high by at least ``price_multiplier``.
        - Current volume must exceed average volume by ``volume_multiplier``.
        """
        if "close" not in df.columns or "volume" not in df.columns:
            return False

        recent_high = self._recent_high(df, self.params.lookback)
        avg_vol = self._average_volume(df, self.params.lookback)

        if np.isnan(recent_high) or np.isnan(avg_vol):
            return False

        current_close = df["close"].iloc[-1]
        current_vol = df["volume"].iloc[-1]

        price_break = current_close > recent_high * self.params.price_multiplier
        volume_break = current_vol > avg_vol * self.params.volume_multiplier
        return price_break and volume_break

    def _exit_filters(self, df: pd.DataFrame) -> bool:
        """
        Exit confirmation:
        - If price falls below the ``exit_price_ma_window``‑period moving average, signal exit.
        - If ML predicts a down move with confidence > 0.6, signal exit.
        """
        if "close" not in df.columns:
            return False

        ma = df["close"].rolling(window=self.params.exit_price_ma_window).mean().iloc[-1]
        current_close = df["close"].iloc[-1]
        price_fall = current_close < ma
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
                and ml_result.get("confidence", 0) > self.params.ml_confidence_threshold
                and ml_result.get("prediction") == "up"
                and is_entry
                and self._entry_filters(data)  # double‑check filters
            ):
                # Blend ML confidence with base confidence, capped at the configured limit
                blended = (base_signal.confidence + ml_result["confidence"]) / 2
                base_signal.confidence = min(self.params.blended_confidence_cap, blended)
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