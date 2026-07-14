"""Ensemble strategy: pure ML signal from all models combined with additional confirmation filters."""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.ml.inference import get_inference_service

# Constants
TICK_INTERVAL_SECONDS = 300.0
CONFIDENCE_THRESHOLD = 0.70
SMA_WINDOW = 20

# Column names
COL_CLOSE = "close"
COL_VOLUME = "volume"
COL_ML_PREDICTION = "ml_prediction"
COL_ML_CONFIDENCE = "ml_confidence"

# Prediction strings
PRED_UP = "up"
PRED_DOWN = "down"
PRED_NEUTRAL = "neutral"

# Side strings
SIDE_BUY = "buy"
SIDE_SELL = "sell"


class EnsembleStrategy(AbstractStrategy):
    name = "ensemble"
    display_name = "Ensemble ML (LSTM + XGB + Lorentzian)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = TICK_INTERVAL_SECONDS
    confidence_threshold = CONFIDENCE_THRESHOLD
    sma_window = SMA_WINDOW

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
            ml_result = await inference.predict(data, symbol)

            # Basic ML validation
            if not ml_result or ml_result.get("prediction") == PRED_NEUTRAL:
                return None
            if ml_result.get("confidence", 0) < self.confidence_threshold:
                return None

            # Ensure we have price and volume data for confirmation
            if COL_CLOSE not in data.columns or COL_VOLUME not in data.columns:
                return None

            # Compute SMA and median volume on the latest slice
            recent = data.tail(self.sma_window)
            if recent.empty:
                return None
            sma = recent[COL_CLOSE].mean()
            median_vol = recent[COL_VOLUME].median()
            latest_close = data[COL_CLOSE].iloc[-1]
            latest_vol = data[COL_VOLUME].iloc[-1]

            # Directional confirmation
            if ml_result["prediction"] == PRED_UP:
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
                side=SIDE_BUY if ml_result["prediction"] == PRED_UP else SIDE_SELL,
                confidence=ml_result["confidence"],
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata=ml_result,
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
        required_cols = {COL_CLOSE, COL_VOLUME, COL_ML_PREDICTION, COL_ML_CONFIDENCE}
        if not required_cols.issubset(df.columns):
            # If required columns are missing, return empty signals to avoid crashes.
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        # Compute rolling SMA and median volume
        sma = df[COL_CLOSE].rolling(window=self.sma_window, min_periods=1).mean()
        median_vol = df[COL_VOLUME].rolling(window=self.sma_window, min_periods=1).median()

        # Conditions for a valid entry
        is_up = df[COL_ML_PREDICTION] == PRED_UP
        is_down = df[COL_ML_PREDICTION] == PRED_DOWN
        conf_ok = df[COL_ML_CONFIDENCE] >= self.confidence_threshold
        price_above_sma = df[COL_CLOSE] > sma
        price_below_sma = df[COL_CLOSE] < sma
        vol_ok = df[COL_VOLUME] >= median_vol

        long_entry = is_up & conf_ok & price_above_sma & vol_ok
        short_entry = is_down & conf_ok & price_below_sma & vol_ok

        entries = long_entry | short_entry

        # Exit when any of the entry conditions become false for the current side.
        # For simplicity we treat the opposite side as an exit signal.
        exit_long = (~price_above_sma) | (~vol_ok) | (df[COL_ML_PREDICTION] == PRED_DOWN)
        exit_short = (~price_below_sma) | (~vol_ok) | (df[COL_ML_PREDICTION] == PRED_UP)
        exits = exit_long | exit_short

        # Align boolean Series with BacktestSignals expectations
        entries = entries.astype(bool)
        exits = exits.astype(bool)

        return BacktestSignals(entries=entries, exits=exits)