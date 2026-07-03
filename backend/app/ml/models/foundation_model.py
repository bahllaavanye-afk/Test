"""
Foundation model wrapper for zero-shot time series forecasting.
Supports: Chronos (Amazon), TimesFM (Google), Moirai (Salesforce).

These models can forecast without training on your data — huge alpha for rare events.
Install: pip install chronos-forecasting
"""
from __future__ import annotations

import numpy as np
import traceback
from typing import Literal

from app.utils.logging import logger

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

try:
    from chronos import ChronosPipeline
    HAS_CHRONOS = True
except ImportError:
    HAS_CHRONOS = False


class FoundationModelSignal:
    """
    Zero-shot trading signal from foundation time series models.
    Uses Chronos-T5-tiny (free, CPU-friendly) by default.
    Falls back to naive baseline if model not installed.
    """

    SUPPORTED = ["chronos-tiny", "chronos-small", "naive"]

    def __init__(self, model_name: Literal["chronos-tiny", "chronos-small", "naive"] = "naive"):
        if model_name not in self.SUPPORTED:
            raise ValueError(f"Unsupported model_name '{model_name}'. Supported: {self.SUPPORTED}")
        self.model_name = model_name
        self._pipeline = None
        self._loaded = False

    def _load(self) -> None:
        """Load the underlying model pipeline if not already loaded."""
        if self._loaded:
            return

        if self.model_name.startswith("chronos") and HAS_CHRONOS:
            size = "tiny" if "tiny" in self.model_name else "small"
            logger.info(
                "Loading Chronos model",
                extra={"model": f"chronos-{size}", "stage": "init"},
            )
            try:
                self._pipeline = ChronosPipeline.from_pretrained(
                    f"amazon/chronos-t5-{size}",
                    device_map="cpu",
                    torch_dtype=torch.float32,
                )
                logger.info(
                    "Chronos model loaded successfully",
                    extra={"model": f"chronos-{size}", "stage": "loaded"},
                )
            except (RuntimeError, OSError) as e:
                logger.error(
                    f"Failed to load Chronos model '{size}': {e}",
                    extra={"model": f"chronos-{size}", "stage": "load_error"},
                )
                logger.debug(traceback.format_exc())
                self.model_name = "naive"
        else:
            if self.model_name != "naive":
                logger.warning(
                    "Chronos not installed. Falling back to naive baseline.",
                    extra={"requested_model": self.model_name},
                )
            self.model_name = "naive"

        self._loaded = True

    def forecast(self, prices: list[float], horizon: int = 5) -> dict:
        """
        Generate price direction forecast.

        Args:
            prices: Historical close prices (minimum 30 points recommended).
            horizon: Number of steps to forecast (must be positive).

        Returns:
            dict with: direction (+1/-1), confidence, quantile forecasts.
        """
        if not isinstance(prices, list) or not all(isinstance(p, (int, float)) for p in prices):
            raise TypeError("prices must be a list of numeric types")
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer")

        self._load()
        arr = np.array(prices, dtype=np.float32)

        if self.model_name == "naive" or not HAS_CHRONOS:
            return self._naive_forecast(arr, horizon)

        # Chronos forecast
        try:
            context = torch.tensor(arr).unsqueeze(0)  # (1, T)
            forecast = self._pipeline.predict(
                context,
                prediction_length=horizon,
                num_samples=20,
            )
            # forecast shape: (num_samples, 1, horizon)
            samples = forecast[0].numpy()  # (num_samples, horizon)
            median = np.median(samples, axis=0)
            q10 = np.percentile(samples, 10, axis=0)
            q90 = np.percentile(samples, 90, axis=0)
            last_price = arr[-1]
            forecast_end = float(np.median(samples[:, -1]))
            direction = 1 if forecast_end > last_price else -1
            confidence = min(abs(forecast_end - last_price) / (last_price * 0.01 + 1e-9), 1.0)

            return {
                "model": "chronos",
                "direction": direction,
                "confidence": round(float(confidence), 3),
                "forecast_median": median.tolist(),
                "forecast_q10": q10.tolist(),
                "forecast_q90": q90.tolist(),
                "horizon": horizon,
            }
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(
                f"Chronos forecast error: {e}",
                extra={"model": self.model_name, "horizon": horizon},
            )
            logger.debug(traceback.format_exc())
            return self._naive_forecast(arr, horizon)
        except Exception as e:  # pragma: no cover
            # Catch-all for unexpected errors; ensure fallback to naive forecast.
            logger.exception(
                f"Unexpected error during Chronos forecasting: {e}",
                extra={"model": self.model_name, "horizon": horizon},
            )
            return self._naive_forecast(arr, horizon)

    def _naive_forecast(self, arr: np.ndarray, horizon: int) -> dict:
        """Simple momentum baseline: 20-day SMA direction."""
        if len(arr) < 20:
            return {
                "model": "naive",
                "direction": 0,
                "confidence": 0.5,
                "horizon": horizon,
            }
        sma20 = np.mean(arr[-20:])
        direction = 1 if arr[-1] > sma20 else -1
        confidence = min(abs(arr[-1] - sma20) / sma20, 0.8)
        forecast_prices = [
            arr[-1] * (1 + direction * 0.001 * i) for i in range(1, horizon + 1)
        ]
        return {
            "model": "naive_momentum",
            "direction": direction,
            "confidence": round(float(confidence), 3),
            "forecast_median": forecast_prices,
            "forecast_q10": [p * 0.99 for p in forecast_prices],
            "forecast_q90": [p * 1.01 for p in forecast_prices],
            "horizon": horizon,
        }


# Module-level singleton (lazy-loaded)
_signal_instance: FoundationModelSignal | None = None


def get_foundation_signal(model_name: str = "naive") -> FoundationModelSignal:
    """
    Retrieve a singleton FoundationModelSignal instance.

    Args:
        model_name: Desired model identifier; must be one of the supported names.

    Returns:
        FoundationModelSignal instance configured with the requested model.
    """
    if model_name not in FoundationModelSignal.SUPPORTED:
        raise ValueError(
            f"Invalid model_name '{model_name}'. Supported models: {FoundationModelSignal.SUPPORTED}"
        )
    global _signal_instance
    if _signal_instance is None or _signal_instance.model_name != model_name:
        _signal_instance = FoundationModelSignal(model_name)
    return _signal_instance