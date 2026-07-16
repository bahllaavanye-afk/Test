"""Ensemble strategy: pure ML signal from all models combined with additional confirmation filters."""
import pandas as pd
import numpy as np
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.ml.inference import get_inference_service


class EnsembleStrategy(AbstractStrategy):
    name = "ensemble"
    display_name = "Ensemble ML (LSTM + XGB + Lorentzian)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0

    # Core thresholds
    confidence_threshold = 0.70          # higher bar for pure ML
    sma_window = 20                       # SMA window for price confirmation
    atr_window = 14                       # ATR window for volatility filter
    max_atr_ratio = 0.02                  # Max ATR / close ratio for entry

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Produce a trading signal based on the ML inference combined with
        price‑based confirmation filters.

        Entry Conditions (tightened)
        ----------------------------
        1. ML model predicts a directional move (up/down) with confidence >= threshold.
        2. Close price is above the SMA for a long signal, below the SMA for a short.
        3. Volume is above the median of the recent window.
        4. Recent price momentum aligns with the prediction (close > previous close for up,
           close < previous close for down).
        5. Current ATR relative to price is below `max_atr_ratio` (low volatility entry).

        Exit Conditions
        ----------------
        An active position is closed when any of the entry conditions become false
        or when the ML prediction flips direction.
        """
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)

            # Basic ML validation
            if not ml_result or ml_result.get("prediction") == "neutral":
                return None
            if ml_result.get("confidence", 0) < self.confidence_threshold:
                return None

            # Required price/volume columns
            required_price_cols = {"close", "volume", "high", "low"}
            if not required_price_cols.issubset(data.columns):
                return None

            # Compute SMA, median volume, ATR and momentum on the recent window
            recent = data.tail(self.sma_window)
            if recent.empty:
                return None

            sma = recent["close"].mean()
            median_vol = recent["volume"].median()
            latest_close = data["close"].iloc[-1]
            latest_vol = data["volume"].iloc[-1]

            # Momentum: compare latest close to previous close
            if len(data) < 2:
                return None
            prev_close = data["close"].iloc[-2]

            # ATR calculation
            tr = pd.concat([
                data["high"] - data["low"],
                (data["high"] - data["close"].shift()).abs(),
                (data["low"] - data["close"].shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=self.atr_window, min_periods=1).mean().iloc[-1]
            atr_ratio = atr / latest_close if latest_close != 0 else np.inf

            # Directional confirmation
            prediction = ml_result["prediction"]
            if prediction == "up":
                if not (latest_close > sma and latest_close > prev_close):
                    return None
            else:  # prediction == "down"
                if not (latest_close < sma and latest_close < prev_close):
                    return None

            # Volume confirmation
            if latest_vol < median_vol:
                return None

            # Volatility confirmation
            if atr_ratio > self.max_atr_ratio:
                return None

            return Signal(
                symbol=symbol,
                side="buy" if prediction == "up" else "sell",
                confidence=ml_result["confidence"],
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata=ml_result,
            )
        except Exception:
            # Production code should log the exception; omitted for brevity.
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Generate entry and exit signals for back‑testing.

        Expected DataFrame columns:
        - 'close', 'volume', 'high', 'low'
        - 'ml_prediction' (string: "up", "down", "neutral")
        - 'ml_confidence' (float 0‑1)
        """
        required_cols = {"close", "volume", "high", "low", "ml_prediction", "ml_confidence"}
        if not required_cols.issubset(df.columns):
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        # Rolling indicators
        sma = df["close"].rolling(window=self.sma_window, min_periods=1).mean()
        median_vol = df["volume"].rolling(window=self.sma_window, min_periods=1).median()
        prev_close = df["close"].shift(1)

        # ATR calculation
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_window, min_periods=1).mean()
        atr_ratio = atr / df["close"].replace(0, np.nan)

        # Core conditions
        conf_ok = df["ml_confidence"] >= self.confidence_threshold
        vol_ok = df["volume"] >= median_vol
        atr_ok = atr_ratio <= self.max_atr_ratio

        # Directional price conditions with momentum
        price_up = (df["close"] > sma) & (df["close"] > prev_close)
        price_down = (df["close"] < sma) & (df["close"] < prev_close)

        is_up = df["ml_prediction"] == "up"
        is_down = df["ml_prediction"] == "down"

        long_entry = is_up & conf_ok & price_up & vol_ok & atr_ok
        short_entry = is_down & conf_ok & price_down & vol_ok & atr_ok

        entries = long_entry | short_entry

        # Exit logic: any condition turning false or prediction flipping
        exit_long = (~price_up) | (~vol_ok) | (~atr_ok) | (df["ml_prediction"] == "down")
        exit_short = (~price_down) | (~vol_ok) | (~atr_ok) | (df["ml_prediction"] == "up")
        exits = exit_long | exit_short

        entries = entries.astype(bool)
        exits = exits.astype(bool)

        return BacktestSignals(entries=entries, exits=exits)