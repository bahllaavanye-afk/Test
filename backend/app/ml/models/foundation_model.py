"""
Foundation model wrapper for zero‑shot time series forecasting.
Supports: Chronos (Amazon), TimesFM (Google), Moirai (Salesforce).

These models can forecast without training on your data — huge alpha for rare events.
Install: pip install chronos-forecasting
"""
from __future__ import annotations

import numpy as np
from typing import Literal

from app.utils.logging import logger

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

try:
    from chronos import ChronosPipeline
    HAS_CHRONOS = True
except ImportError:  # pragma: no cover
    HAS_CHRONOS = False


class FoundationModelSignal:
    """
    Zero‑shot trading signal from foundation time series models.
    Uses Chronos‑T5‑tiny (free, CPU‑friendly) by default.
    Falls back to naive baseline if model not installed.
    """

    SUPPORTED = ["chronos-tiny", "chronos-small", "naive"]

    # Confidence threshold for taking a non‑neutral signal
    _CONFIDENCE_THRESHOLD = 0.6
    # Volatility bounds (as a fraction of price) for accepting a signal
    _VOLATILITY_MIN = 0.001
    _VOLATILITY_MAX = 0.05
    # SMA windows for confirmation filters
    _SMA_SHORT = 20
    _SMA_LONG = 50

    def __init__(self, model_name: Literal["chronos-tiny", "chronos-small", "naive"] = "naive"):
        self.model_name = model_name
        self._pipeline = None
        self._loaded = False

    # --------------------------------------------------------------------- #
    # Loading utilities
    # --------------------------------------------------------------------- #
    def _load(self) -> None:
        if self._loaded:
            return
        if self.model_name.startswith("chronos") and HAS_CHRONOS:
            size = "tiny" if "tiny" in self.model_name else "small"
            logger.info(f"Loading Chronos {size} model (first load may download weights)...")
            self._pipeline = ChronosPipeline.from_pretrained(
                f"amazon/chronos-t5-{size}",
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            logger.info("Chronos loaded.")
        else:
            if self.model_name != "naive":
                logger.warning(
                    "chronos not installed. Using naive baseline. pip install chronos-forecasting"
                )
            self.model_name = "naive"
        self._loaded = True

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def forecast(self, prices: list[float], horizon: int = 5) -> dict:
        """
        Generate price direction forecast.

        Args:
            prices: Historical close prices (min 30)
            horizon: Number of steps to forecast

        Returns:
            dict with direction (+1/-1/0), confidence, quantile forecasts,
            and suggested exit parameters (stop_loss, take_profit).
        """
        self._load()
        arr = np.array(prices, dtype=np.float32)

        if self.model_name == "naive" or not HAS_CHRONOS:
            return self._naive_forecast(arr, horizon)

        # Chronos forecast
        try:
            context = torch.tensor(arr).unsqueeze(0)  # (1, T)
            forecast = self._pipeline.predict(
                context, prediction_length=horizon, num_samples=20
            )
            # forecast shape: (num_samples, 1, horizon)
            samples = forecast[0].numpy()  # (num_samples, horizon)
            median = np.median(samples, axis=0)
            q10 = np.percentile(samples, 10, axis=0)
            q90 = np.percentile(samples, 90, axis=0)

            last_price = arr[-1]
            forecast_end = float(np.median(samples[:, -1]))
            direction = 1 if forecast_end > last_price else -1
            confidence = min(
                abs(forecast_end - last_price) / (last_price * 0.01 + 1e-9), 1.0
            )

            # Apply confirmation filters
            direction, confidence = self._apply_entry_filters(
                arr, direction, confidence
            )

            # Build exit suggestions from forecast quantiles
            stop_loss, take_profit = self._derive_exit_levels(last_price, q10, q90, direction)

            return {
                "model": "chronos",
                "direction": direction,
                "confidence": round(float(confidence), 3),
                "forecast_median": median.tolist(),
                "forecast_q10": q10.tolist(),
                "forecast_q90": q90.tolist(),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "horizon": horizon,
            }
        except Exception as e:  # pragma: no cover
            logger.error(f"Chronos forecast error: {e}")
            return self._naive_forecast(arr, horizon)

    # --------------------------------------------------------------------- #
    # Naive baseline
    # --------------------------------------------------------------------- #
    def _naive_forecast(self, arr: np.ndarray, horizon: int) -> dict:
        """Simple momentum baseline with added confirmation filters."""
        if len(arr) < self._SMA_LONG:
            # Not enough data – return neutral signal
            return {
                "model": "naive",
                "direction": 0,
                "confidence": 0.5,
                "forecast_median": [],
                "forecast_q10": [],
                "forecast_q90": [],
                "stop_loss": None,
                "take_profit": None,
                "horizon": horizon,
            }

        sma20 = np.mean(arr[-self._SMA_SHORT :])
        sma50 = np.mean(arr[-self._SMA_LONG :])
        direction = 1 if arr[-1] > sma20 else -1
        confidence = min(abs(arr[-1] - sma20) / sma20, 0.8)

        # Confirmation: direction must agree with longer SMA trend
        if direction == 1 and arr[-1] < sma50:
            direction = 0
            confidence = 0.4
        elif direction == -1 and arr[-1] > sma50:
            direction = 0
            confidence = 0.4

        # Confidence threshold
        if confidence < self._CONFIDENCE_THRESHOLD:
            direction = 0

        # Generate a simple linear forecast
        forecast_prices = [
            arr[-1] * (1 + direction * 0.001 * i) for i in range(1, horizon + 1)
        ]
        q10 = [p * 0.99 for p in forecast_prices]
        q90 = [p * 1.01 for p in forecast_prices]

        stop_loss, take_profit = self._derive_exit_levels(
            arr[-1], q10, q90, direction
        )

        return {
            "model": "naive_momentum",
            "direction": direction,
            "confidence": round(float(confidence), 3),
            "forecast_median": forecast_prices,
            "forecast_q10": q10,
            "forecast_q90": q90,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "horizon": horizon,
        }

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def _apply_entry_filters(
        self, arr: np.ndarray, direction: int, confidence: float
    ) -> tuple[int, float]:
        """
        Tighten entry conditions using volatility and longer‑term trend filters.

        Returns possibly adjusted (direction, confidence).
        """
        # Volatility filter – reject extreme or negligible volatility
        recent_vol = self._recent_volatility(arr)
        if recent_vol < self._VOLATILITY_MIN or recent_vol > self._VOLATILITY_MAX:
            return 0, confidence * 0.5

        # Longer SMA confirmation
        sma_long = np.mean(arr[-self._SMA_LONG :])
        last_price = arr[-1]
        if direction == 1 and last_price < sma_long:
            direction = 0
            confidence *= 0.5
        elif direction == -1 and last_price > sma_long:
            direction = 0
            confidence *= 0.5

        # Confidence threshold
        if confidence < self._CONFIDENCE_THRESHOLD:
            direction = 0

        return direction, confidence

    def _recent_volatility(self, arr: np.ndarray, window: int = 10) -> float:
        """
        Compute recent volatility as standard deviation divided by mean price.
        """
        if len(arr) < window:
            return 0.0
        recent = arr[-window:]
        std = np.std(recent)
        mean = np.mean(recent)
        return std / (mean + 1e-9)

    def _derive_exit_levels(
        self,
        last_price: float,
        q10: list[float],
        q90: list[float],
        direction: int,
    ) -> tuple[float | None, float | None]:
        """
        Provide simple stop‑loss and take‑profit levels based on forecast quantiles.
        """
        if direction == 0:
            return None, None

        # Use the first‑step quantiles as a proxy for risk/reward
        sl = q10[0] if direction == 1 else q90[0]
        tp = q90[0] if direction == 1 else q10[0]

        # Ensure stop‑loss is not beyond last price in the wrong direction
        if direction == 1 and sl > last_price:
            sl = last_price * 0.99
        if direction == -1 and sl < last_price:
            sl = last_price * 1.01

        return round(float(sl), 4), round(float(tp), 4)


# Module-level singleton (lazy‑loaded)
_signal_instance: FoundationModelSignal | None = None


def get_foundation_signal(model_name: str = "naive") -> FoundationModelSignal:
    global _signal_instance
    if _signal_instance is None or _signal_instance.model_name != model_name:
        _signal_instance = FoundationModelSignal(model_name)
    return _signal_instance