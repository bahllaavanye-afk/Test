"""ML-filtered mean reversion. Reduces false signals by tightening entry criteria and adding confirmation filters."""
import pandas as pd
import logging
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.mean_reversion import MeanReversionStrategy
from app.ml.inference import get_inference_service

logger = logging.getLogger(__name__)


class MLMeanReversionStrategy(AbstractStrategy):
    name = "ml_mean_reversion"
    display_name = "ML Mean Reversion (BB + ML Filter)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._base = MeanReversionStrategy(params)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Produce a signal only when:
        1. The underlying mean‑reversion strategy generates a signal.
        2. The price is sufficiently away from its Bollinger Bands (strong reversion potential).
        3. Recent volume is above the 20‑period average (liquidity confirmation).
        4. The ML model predicts the same direction with confidence > 0.70.
        The signal confidence is then modestly boosted.
        """
        # Basic sanity checks
        required_cols = {"close", "volume"}
        if not required_cols.issubset(data.columns):
            logger.debug("Missing required columns for MLMeanReversionStrategy: %s", required_cols - set(data.columns))
            return None
        if len(data) < 30:
            logger.debug("Insufficient data length (%d rows) for analysis", len(data))
            return None

        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None

        # Bollinger Bands confirmation
        close_series = data["close"]
        rolling_mean = close_series.rolling(window=20, min_periods=20).mean()
        rolling_std = close_series.rolling(window=20, min_periods=20).std()
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std
        latest_price = close_series.iloc[-1]

        # For a mean‑reversion entry we expect price beyond the bands opposite to the trade side
        if base_signal.side == "buy" and latest_price > upper_band.iloc[-1]:
            price_confirm = True
        elif base_signal.side == "sell" and latest_price < lower_band.iloc[-1]:
            price_confirm = True
        else:
            logger.debug(
                "Bollinger band condition not met for %s signal on %s (price=%.2f, bands=[%.2f, %.2f])",
                base_signal.side,
                symbol,
                latest_price,
                lower_band.iloc[-1],
                upper_band.iloc[-1],
            )
            return None

        # Volume confirmation (average of last 20 periods)
        avg_volume = data["volume"].rolling(window=20, min_periods=20).mean().iloc[-1]
        latest_volume = data["volume"].iloc[-1]
        if latest_volume < 1.2 * avg_volume:
            logger.debug(
                "Volume condition not met for %s signal on %s (latest=%.2f, avg=%.2f)",
                base_signal.side,
                symbol,
                latest_volume,
                avg_volume,
            )
            return None

        # ML prediction confirmation
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if not ml_result:
                logger.debug("ML inference returned no result for %s", symbol)
                return None

            ml_confidence = ml_result.get("confidence", 0.0)
            ml_prediction = ml_result.get("prediction", "").lower()

            if ml_confidence < 0.70:
                logger.debug("ML confidence %.2f below threshold for %s", ml_confidence, symbol)
                return None

            direction_match = (
                (ml_prediction == "up" and base_signal.side == "buy")
                or (ml_prediction == "down" and base_signal.side == "sell")
            )
            if not direction_match:
                logger.debug(
                    "ML prediction '%s' disagrees with base signal side '%s' for %s",
                    ml_prediction,
                    base_signal.side,
                    symbol,
                )
                return None
        except Exception as exc:
            logger.exception("ML inference error for %s: %s", symbol, exc)
            # Fallback to base signal if ML fails, but keep existing confirmations
            # (price and volume already satisfied)
            pass
        else:
            # Boost confidence modestly, capped to avoid over‑confidence
            base_signal.confidence = min(0.93, base_signal.confidence * 1.07)

        # Attach strategy metadata
        base_signal.strategy_name = self.name
        base_signal.strategy_type = self.strategy_type

        # Optional exit improvement: set a dynamic stop‑loss based on recent volatility
        if hasattr(base_signal, "stop_loss") and hasattr(base_signal, "take_profit"):
            # Use 1.5 * recent std dev as stop‑loss distance
            recent_std = rolling_std.iloc[-1]
            if base_signal.side == "buy":
                base_signal.stop_loss = latest_price - 1.5 * recent_std
                base_signal.take_profit = latest_price + 3 * recent_std
            else:
                base_signal.stop_loss = latest_price + 1.5 * recent_std
                base_signal.take_profit = latest_price - 3 * recent_std

        return base_signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        # Reuse the underlying strategy's backtest implementation
        return self._base.backtest_signals(df)