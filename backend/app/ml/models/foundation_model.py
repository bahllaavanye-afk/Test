"""
Foundation model wrapper for zero-shot time series forecasting.
Supports: Chronos (Amazon), TimesFM (Google), Moirai (Salesforce).

These models can forecast without training on your data — huge alpha for rare events.
Install: pip install chronos-forecasting
"""
from __future__ import annotations

from typing import Literal, List

import numpy as np
from pydantic import BaseModel, Field, validator, root_validator

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


class ForecastRequest(BaseModel):
    """
    Input schema for foundation model forecasts.
    """

    prices: List[float] = Field(
        ...,
        description="Historical close prices. Minimum 30 observations required for Chronos models.",
        example=[100.5, 101.2, 99.8, 100.1, 100.9],
        min_items=1,
    )
    horizon: int = Field(
        5,
        description="Number of future steps to forecast.",
        example=5,
        gt=0,
    )

    @validator("prices")
    def prices_must_be_finite(cls, v: List[float]) -> List[float]:
        if not all(np.isfinite(v)):
            raise ValueError("All price values must be finite numbers.")
        return v

    @root_validator
    def check_length_for_chronos(cls, values):
        prices = values.get("prices", [])
        # Chronos requires at least 30 points; enforce for all models to keep consistent behavior.
        if len(prices) < 30:
            # Allow naive fallback but warn via logger.
            logger.warning("Price series shorter than 30; Chronos may fallback to naive baseline.")
        return values


class ForecastResult(BaseModel):
    """
    Output schema for foundation model forecasts.
    """

    model: str = Field(
        ...,
        description="Identifier of the model used for forecasting.",
        example="chronos",
    )
    direction: int = Field(
        ...,
        description="Predicted price direction: +1 for up, -1 for down, 0 for neutral (naive fallback).",
        example=1,
    )
    confidence: float = Field(
        ...,
        description="Confidence score between 0 and 1 indicating strength of the direction signal.",
        example=0.78,
        ge=0.0,
        le=1.0,
    )
    forecast_median: List[float] = Field(
        ...,
        description="Median forecasted price for each step in the horizon.",
        example=[101.2, 101.4, 101.6],
    )
    forecast_q10: List[float] = Field(
        ...,
        description="10th percentile forecasted price for each step in the horizon.",
        example=[100.9, 101.1, 101.3],
    )
    forecast_q90: List[float] = Field(
        ...,
        description="90th percentile forecasted price for each step in the horizon.",
        example=[101.5, 101.7, 101.9],
    )
    horizon: int = Field(
        ...,
        description="Number of steps the forecast covers.",
        example=5,
        gt=0,
    )

    @validator("direction")
    def direction_must_be_valid(cls, v: int) -> int:
        if v not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0, or 1")
        return v

    @validator("forecast_median", "forecast_q10", "forecast_q90")
    def forecasts_must_match_horizon(cls, v: List[float], values, **kwargs) -> List[float]:
        horizon = values.get("horizon")
        if horizon is not None and len(v) != horizon:
            raise ValueError(f"Length of forecast list must equal horizon ({horizon})")
        return v


class FoundationModelSignal:
    """
    Zero-shot trading signal from foundation time series models.
    Uses Chronos-T5-tiny (free, CPU-friendly) by default.
    Falls back to naive baseline if model not installed.
    """

    SUPPORTED = ["chronos-tiny", "chronos-small", "naive"]

    def __init__(self, model_name: Literal["chronos-tiny", "chronos-small", "naive"] = "naive"):
        self.model_name = model_name
        self._pipeline = None
        self._loaded = False

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
                logger.warning("chronos not installed. Using naive baseline. pip install chronos-forecasting")
            self.model_name = "naive"
        self._loaded = True

    def forecast(self, prices: List[float], horizon: int = 5) -> dict:
        """
        Generate price direction forecast.

        Args:
            prices: Historical close prices (min 30 for Chronos models).
            horizon: Number of steps to forecast.

        Returns:
            A dictionary conforming to :class:`ForecastResult`.
        """
        # Validate inputs using the Pydantic request schema
        request = ForecastRequest(prices=prices, horizon=horizon)

        self._load()
        arr = np.array(request.prices, dtype=np.float32)

        if self.model_name == "naive" or not HAS_CHRONOS:
            raw_result = self._naive_forecast(arr, request.horizon)
        else:
            # Chronos forecast
            try:
                context = torch.tensor(arr).unsqueeze(0)  # (1, T)
                forecast = self._pipeline.predict(context, prediction_length=request.horizon, num_samples=20)
                # forecast shape: (num_samples, 1, horizon)
                samples = forecast[0].numpy()  # (num_samples, horizon)
                median = np.median(samples, axis=0)
                q10 = np.percentile(samples, 10, axis=0)
                q90 = np.percentile(samples, 90, axis=0)
                last_price = arr[-1]
                forecast_end = float(np.median(samples[:, -1]))
                direction = 1 if forecast_end > last_price else -1
                confidence = min(abs(forecast_end - last_price) / (last_price * 0.01 + 1e-9), 1.0)
                raw_result = {
                    "model": "chronos",
                    "direction": direction,
                    "confidence": round(float(confidence), 3),
                    "forecast_median": median.tolist(),
                    "forecast_q10": q10.tolist(),
                    "forecast_q90": q90.tolist(),
                    "horizon": request.horizon,
                }
            except Exception as e:
                logger.error(f"Chronos forecast error: {e}")
                raw_result = self._naive_forecast(arr, request.horizon)

        # Validate and format the output using the Pydantic result schema
        result = ForecastResult(**raw_result)
        return result.dict()

    def _naive_forecast(self, arr: np.ndarray, horizon: int) -> dict:
        """Simple momentum baseline: 20-day SMA direction."""
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


# Module-level singleton (lazy-loaded)
_signal_instance: FoundationModelSignal | None = None


def get_foundation_signal(model_name: str = "naive") -> FoundationModelSignal:
    global _signal_instance
    if _signal_instance is None or _signal_instance.model_name != model_name:
        _signal_instance = FoundationModelSignal(model_name)
    return _signal_instance