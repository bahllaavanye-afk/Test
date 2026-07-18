"""
Foundation model wrapper for zero‑shot time series forecasting.
Supports: Chronos (Amazon), TimesFM (Google), Moirai (Salesforce).

These models can forecast without training on your data — huge alpha for rare events.
Install: pip install chronos-forecasting
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Literal

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
    Zero‑shot trading signal generator using foundation time‑series models.

    The default model is ``chronos‑tiny`` which runs on CPU and requires no
    additional training. If Chronos is unavailable, the class falls back to a
    simple naive momentum baseline.

    Attributes
    ----------
    model_name: Literal["chronos-tiny", "chronos-small", "naive"]
        Identifier of the underlying forecasting model.
    _pipeline: ChronosPipeline | None
        Loaded Chronos pipeline; initialised lazily on first use.
    _loaded: bool
        Flag indicating whether the model has been loaded.
    """

    SUPPORTED = ["chronos-tiny", "chronos-small", "naive"]

    def __init__(self, model_name: Literal["chronos-tiny", "chronos-small", "naive"] = "naive") -> None:
        """
        Initialise the signal wrapper.

        Parameters
        ----------
        model_name: Literal["chronos-tiny", "chronos-small", "naive"], optional
            Desired model; defaults to ``"naive"`` when Chronos is not installed.
        """
        self.model_name = model_name
        self._pipeline = None
        self._loaded = False

    def _load(self) -> None:
        """
        Lazily load the selected model.

        If a Chronos model is requested and the package is available, the
        corresponding pretrained weights are downloaded (if needed) and a
        ``ChronosPipeline`` instance is created. Otherwise the wrapper switches
        to the naive baseline.
        """
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

    def forecast(self, prices: list[float], horizon: int = 5) -> Dict[str, Any]:
        """
        Generate a price‑direction forecast for the given price history.

        Parameters
        ----------
        prices: list[float]
            Historical close prices; at least 30 points are recommended.
        horizon: int, optional
            Number of future steps to forecast; defaults to ``5``.

        Returns
        -------
        dict
            A dictionary containing:
            - ``model``: identifier of the model used.
            - ``direction``: ``+1`` for upward, ``-1`` for downward, ``0`` if neutral.
            - ``confidence``: confidence score between 0 and 1.
            - ``forecast_median``: median forecast values for each horizon step.
            - ``forecast_q10`` / ``forecast_q90``: 10th and 90th percentile forecasts.
            - ``horizon``: the forecast horizon supplied.
        """
        self._load()
        arr = np.array(prices, dtype=np.float32)

        if self.model_name == "naive" or not HAS_CHRONOS:
            return self._naive_forecast(arr, horizon)

        # Chronos forecast
        try:
            context = torch.tensor(arr).unsqueeze(0)  # (1, T)
            forecast = self._pipeline.predict(context, prediction_length=horizon, num_samples=20)
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
        except Exception as e:
            logger.error(f"Chronos forecast error: {e}")
            return self._naive_forecast(arr, horizon)

    def _naive_forecast(self, arr: np.ndarray, horizon: int) -> Dict[str, Any]:
        """
        Simple momentum baseline using a 20‑day simple moving average.

        Parameters
        ----------
        arr: np.ndarray
            Numpy array of historical prices.
        horizon: int
            Forecast horizon.

        Returns
        -------
        dict
            Forecast dictionary mirroring the structure of :meth:`forecast`.
        """
        if len(arr) < 20:
            return {"model": "naive", "direction": 0, "confidence": 0.5, "horizon": horizon}
        sma20 = np.mean(arr[-20:])
        direction = 1 if arr[-1] > sma20 else -1
        confidence = min(abs(arr[-1] - sma20) / sma20, 0.8)
        forecast_prices = [arr[-1] * (1 + direction * 0.001 * i) for i in range(1, horizon + 1)]
        return {
            "model": "naive_momentum",
            "direction": direction,
            "confidence": round(float(confidence), 3),
            "forecast_median": forecast_prices,
            "forecast_q10": [p * 0.99 for p in forecast_prices],
            "forecast_q90": [p * 1.01 for p in forecast_prices],
            "horizon": horizon,
        }


# Module-level singleton (lazy‑loaded)
_signal_instance: FoundationModelSignal | None = None


def get_foundation_signal(model_name: str = "naive") -> FoundationModelSignal:
    """
    Retrieve a singleton instance of :class:`FoundationModelSignal`.

    The instance is recreated when the requested ``model_name`` differs from the
    currently cached one.

    Parameters
    ----------
    model_name: str, optional
        Desired model identifier; defaults to ``"naive"``.

    Returns
    -------
    FoundationModelSignal
        The (potentially newly created) signal wrapper.
    """
    global _signal_instance
    if _signal_instance is None or _signal_instance.model_name != model_name:
        _signal_instance = FoundationModelSignal(model_name)
    return _signal_instance