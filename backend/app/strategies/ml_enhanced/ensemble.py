"""Ensemble strategy: pure ML signal from all models combined with additional confirmation filters."""
import logging
from typing import Optional

import pandas as pd
from app.ml.inference import get_inference_service
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

logger = logging.getLogger(__name__)


class EnsembleStrategy(AbstractStrategy):
    name = "ensemble"
    display_name = "Ensemble ML (LSTM + XGB + Lorentzian)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0

    # Core thresholds
    confidence_threshold = 0.70  # higher bar for pure ML
    sma_window = 20  # simple moving average window for confirmation

    # Additional confirmation parameters
    momentum_window = 5
    volatility_multiplier = 0.5
    volume_multiplier = 1.10  # require volume > 10% above median

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Produce a trading signal based on the ML inference combined with
        tighter price‑based confirmation filters.

        Entry Conditions (all must hold):
        1. ML model predicts a directional move (up/down) with confidence >= threshold.
        2. Current close price is above the SMA for a long, below the SMA for a short.
        3. Recent price momentum aligns with the prediction (price change > 0 for long,
           < 0 for short) over ``momentum_window`` periods.
        4. Absolute price change exceeds ``volatility_multiplier`` × recent price volatility
           (rolling std dev).
        5. Current volume exceeds ``volume_multiplier`` × median volume of the SMA window.

        Exit Conditions:
        - Any entry condition fails for the active side.
        - Additionally, a reversal in price relative to SMA triggers an exit.
        """
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)

            # Basic ML validation
            if not ml_result or ml_result.get("prediction") == "neutral":
                return None
            if ml_result.get("confidence", 0) < self.confidence_threshold:
                return None

            # Ensure required columns exist
            required_cols = {"close", "volume"}
            if not required_cols.issubset(data.columns):
                return None

            # Recent slice for SMA, volume, volatility, and momentum
            recent = data.tail(self.sma_window)
            if recent.empty:
                return None

            sma = recent["close"].mean()
            median_vol = recent["volume"].median()
            volatility = recent["close"].rolling(window=self.sma_window, min_periods=1).std().iloc[-1]

            latest_close = data["close"].iloc[-1]
            latest_vol = data["volume"].iloc[-1]

            # Price‑SMA confirmation
            prediction = ml_result["prediction"]
            if prediction == "up" and latest_close <= sma:
                return None
            if prediction == "down" and latest_close >= sma:
                return None

            # Volume confirmation (strict)
            if latest_vol < median_vol * self.volume_multiplier:
                return None

            # Momentum confirmation
            if len(data) >= self.momentum_window + 1:
                past_close = data["close"].iloc[-self.momentum_window - 1]
                price_change = latest_close - past_close
                if prediction == "up" and price_change <= 0:
                    return None
                if prediction == "down" and price_change >= 0:
                    return None
            else:
                # Not enough data for momentum check
                return None

            # Volatility filter
            if volatility > 0:
                if abs(price_change) < volatility * self.volatility_multiplier:
                    return None
            else:
                # Zero volatility (flat market) – be conservative
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
        except Exception as exc:  # pragma: no cover
            logger.exception("Error in EnsembleStrategy.analyze for %s: %s", symbol, exc)
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Generate entry and exit signals for back‑testing, mirroring the runtime logic.

        Expected DataFrame columns:
        - 'close': price series
        - 'volume': volume series
        - 'ml_prediction': string ("up", "down", "neutral")
        - 'ml_confidence': float (0‑1)
        """
        required_cols = {"close", "volume", "ml_prediction", "ml_confidence"}
        if not required_cols.issubset(df.columns):
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        # Rolling calculations
        sma = df["close"].rolling(window=self.sma_window, min_periods=1).mean()
        median_vol = df["volume"].rolling(window=self.sma_window, min_periods=1).median()
        volatility = df["close"].rolling(window=self.sma_window, min_periods=1).std()

        # Momentum over the defined window
        momentum = df["close"].diff(self.momentum_window)

        # Core conditions
        conf_ok = df["ml_confidence"] >= self.confidence_threshold
        price_above_sma = df["close"] > sma
        price_below_sma = df["close"] < sma
        vol_ok = df["volume"] > median_vol * self.volume_multiplier
        momentum_up = momentum > 0
        momentum_down = momentum < 0
        volatilty_ok = (volatility * self.volatility_multiplier) <= momentum.abs()

        # Long entry
        long_entry = (
            (df["ml_prediction"] == "up")
            & conf_ok
            & price_above_sma
            & vol_ok
            & momentum_up
            & volatilty_ok
        )
        # Short entry
        short_entry = (
            (df["ml_prediction"] == "down")
            & conf_ok
            & price_below_sma
            & vol_ok
            & momentum_down
            & volatilty_ok
        )

        entries = long_entry | short_entry

        # Exit logic: opposite SMA breach, loss of volume, or reversal of ML prediction
        exit_long = (~price_above_sma) | (~vol_ok) | (df["ml_prediction"] == "down")
        exit_short = (~price_below_sma) | (~vol_ok) | (df["ml_prediction"] == "up")
        exits = exit_long | exit_short

        entries = entries.astype(bool)
        exits = exits.astype(bool)

        return BacktestSignals(entries=entries, exits=exits)