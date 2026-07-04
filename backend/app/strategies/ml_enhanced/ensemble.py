"""Ensemble strategy: pure ML signal from all models combined with additional confirmation filters."""
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, validator

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.ml.inference import get_inference_service


class MLResult(BaseModel):
    """
    Schema representing the output from the ML inference service.

    Attributes
    ----------
    prediction : str
        Predicted market direction. Must be one of ``'up'``, ``'down'``, or ``'neutral'``.
        Example: ``'up'``.
    confidence : float
        Confidence score of the prediction, ranging from 0.0 to 1.0.
        Example: ``0.87``.
    """

    prediction: str = Field(
        ...,
        description="Predicted market direction. Must be 'up', 'down', or 'neutral'.",
        example="up",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level of the prediction, between 0 and 1.",
        example=0.87,
    )

    class Config:
        extra = "allow"

    @validator("prediction")
    def validate_prediction(cls, v: str) -> str:
        allowed = {"up", "down", "neutral"}
        if v not in allowed:
            raise ValueError(f"prediction must be one of {allowed}")
        return v


class EnsembleConfig(BaseModel):
    """
    Configuration parameters for the Ensemble strategy.

    Attributes
    ----------
    confidence_threshold : float
        Minimum confidence required from the ML model to consider a signal.
        Must be between 0 and 1. Example: ``0.70``.
    sma_window : int
        Window size (in periods) for the simple moving average used in confirmation.
        Must be a positive integer. Example: ``20``.
    """

    confidence_threshold: float = Field(
        0.70,
        ge=0.0,
        le=1.0,
        description="Minimum confidence required from the ML model to emit a signal.",
        example=0.70,
    )
    sma_window: int = Field(
        20,
        ge=1,
        description="Window size for the simple moving average confirmation filter.",
        example=20,
    )


class EnsembleStrategy(AbstractStrategy):
    name = "ensemble"
    display_name = "Ensemble ML (LSTM + XGB + Lorentzian)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0
    confidence_threshold = 0.70  # higher bar for pure ML
    sma_window = 20  # simple moving average window for confirmation

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Produce a trading signal based on the ML inference combined with
        price‑based confirmation filters.

        Entry Conditions
        ----------------
        1. ML model predicts a directional move (up/down) with confidence >= threshold.
        2. Current close price is above the SMA for a long signal, or below the SMA for a short.
        3. Volume is above the median of the recent window (default 20 periods).

        Exit Conditions
        ----------------
        A signal is not emitted if any of the above conditions fail, which the
        back‑testing engine interprets as an exit for the active position.
        """
        try:
            inference = get_inference_service()
            raw_result = await inference.predict(data, symbol)

            # Validate ML output against schema
            try:
                ml_result = MLResult.parse_obj(raw_result)
            except ValidationError:
                return None

            # Basic ML validation
            if ml_result.prediction == "neutral":
                return None
            if ml_result.confidence < self.confidence_threshold:
                return None

            # Ensure we have price and volume data for confirmation
            if "close" not in data.columns or "volume" not in data.columns:
                return None

            # Compute SMA and median volume on the latest slice
            recent = data.tail(self.sma_window)
            if recent.empty:
                return None
            sma = recent["close"].mean()
            median_vol = recent["volume"].median()
            latest_close = data["close"].iloc[-1]
            latest_vol = data["volume"].iloc[-1]

            # Directional confirmation
            if ml_result.prediction == "up":
                if latest_close <= sma:
                    return None
            else:  # prediction == "down"
                if latest_close >= sma:
                    return None

            # Volume confirmation
            if latest_vol < median_vol:
                return None

            return Signal(
                symbol=symbol,
                side="buy" if ml_result.prediction == "up" else "sell",
                confidence=ml_result.confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata=raw_result,
            )
        except Exception:
            # In production we would log the exception; for now we silently ignore.
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Generate entry and exit signals for back‑testing.

        Expected DataFrame columns:
        - 'close': price series
        - 'volume': volume series
        - 'ml_prediction': string ("up", "down", "neutral")
        - 'ml_confidence': float (0‑1)

        The method mirrors the runtime `analyze` logic but operates row‑wise.
        """
        required_cols = {"close", "volume", "ml_prediction", "ml_confidence"}
        if not required_cols.issubset(df.columns):
            # If required columns are missing, return empty signals to avoid crashes.
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        # Compute rolling SMA and median volume
        sma = df["close"].rolling(window=self.sma_window, min_periods=1).mean()
        median_vol = df["volume"].rolling(window=self.sma_window, min_periods=1).median()

        # Conditions for a valid entry
        is_up = df["ml_prediction"] == "up"
        is_down = df["ml_prediction"] == "down"
        conf_ok = df["ml_confidence"] >= self.confidence_threshold
        price_above_sma = df["close"] > sma
        price_below_sma = df["close"] < sma
        vol_ok = df["volume"] >= median_vol

        long_entry = is_up & conf_ok & price_above_sma & vol_ok
        short_entry = is_down & conf_ok & price_below_sma & vol_ok

        entries = long_entry | short_entry

        # Exit when any of the entry conditions become false for the current side.
        # For simplicity we treat the opposite side as an exit signal.
        exit_long = (~price_above_sma) | (~vol_ok) | (df["ml_prediction"] == "down")
        exit_short = (~price_below_sma) | (~vol_ok) | (df["ml_prediction"] == "up")
        exits = exit_long | exit_short

        # Align boolean Series with BacktestSignals expectations
        entries = entries.astype(bool)
        exits = exits.astype(bool)

        return BacktestSignals(entries=entries, exits=exits)