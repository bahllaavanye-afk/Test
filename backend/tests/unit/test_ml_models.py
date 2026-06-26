"""Unit tests for TFT, LightGBM, and Foundation model."""
import numpy as np
import pytest
from pydantic import BaseModel, Field, validator
from typing import List

torch = pytest.importorskip("torch")  # skip module when optional [ml] extra (torch) is absent
from app.ml.models.transformer import TFTModel
from app.ml.models.foundation_model import FoundationModelSignal, get_foundation_signal


class ForecastResult(BaseModel):
    """Schema representing a forecast result from a foundation model.

    Attributes
    ----------
    model: str
        Identifier of the model that generated the forecast.
    direction: int
        Predicted direction: -1 (down), 0 (neutral), or 1 (up).
    confidence: float
        Confidence score in the range [0, 1].
    forecast_median: List[float]
        Median forecast values for each step in the horizon.
    forecast_q10: List[float]
        10th percentile forecast values for each step in the horizon.
    forecast_q90: List[float]
        90th percentile forecast values for each step in the horizon.
    """

    model: str = Field(..., description="Name of the model used for forecasting", example="naive")
    direction: int = Field(..., description="Predicted direction: -1 (down), 0 (neutral), 1 (up)", example=1)
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the forecast, between 0 and 1",
        example=0.85,
    )
    forecast_median: List[float] = Field(
        ..., description="Median forecast values for each horizon step", example=[100.1, 100.2, 100.3]
    )
    forecast_q10: List[float] = Field(
        ..., description="10th percentile forecast values", example=[99.9, 100.0, 100.1]
    )
    forecast_q90: List[float] = Field(
        ..., description="90th percentile forecast values", example=[100.3, 100.4, 100.5]
    )

    @validator("direction")
    def direction_must_be_valid(cls, v):
        if v not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0, or 1")
        return v

    @validator("forecast_median", "forecast_q10", "forecast_q90")
    def lists_must_match_lengths(cls, v, values, **kwargs):
        # Ensure all forecast lists have the same length
        other_keys = {"forecast_median", "forecast_q10", "forecast_q90"} - {kwargs["field"].name}
        lengths = [len(v)]
        for key in other_keys:
            if key in values:
                lengths.append(len(values[key]))
        if len(set(lengths)) > 1:
            raise ValueError("All forecast lists must have the same length")
        return v


class TestTFTModel:
    def _make_batch(self, batch=4, seq=30, features=10):
        return torch.randn(batch, seq, features)

    def test_forward_shape(self):
        model = TFTModel(n_features=10, d_model=32, n_heads=2, seq_len=30)
        x = self._make_batch()
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_bounded_0_1(self):
        model = TFTModel(n_features=10, d_model=32, n_heads=2, seq_len=30)
        x = self._make_batch()
        out = model(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_attention_weights_accessible(self):
        model = TFTModel(n_features=10, d_model=32, n_heads=2, seq_len=30)
        x = self._make_batch()
        model(x)
        weights = model.get_attention_weights()
        assert weights is not None

    def test_predict_proba(self):
        model = TFTModel(n_features=10, d_model=32, n_heads=2, seq_len=30)
        x = self._make_batch()
        probs = model.predict_proba(x)
        assert len(probs) == 4
        assert all(0 <= p <= 1 for p in probs)


class TestFoundationModelSignal:
    def test_naive_forecast(self):
        sig = FoundationModelSignal("naive")
        prices = [100 + i * 0.1 for i in range(50)]
        result = sig.forecast(prices, horizon=5)
        # Validate schema
        ForecastResult(**result)
        assert result["direction"] in (-1, 1)
        assert 0 <= result["confidence"] <= 1
        assert len(result["forecast_median"]) == 5

    def test_short_prices_returns_zero_direction(self):
        sig = FoundationModelSignal("naive")
        result = sig.forecast([100, 101, 102], horizon=3)
        # Validate schema (direction should be 0)
        ForecastResult(**result)
        assert result["direction"] == 0

    def test_get_foundation_signal_singleton(self):
        s1 = get_foundation_signal("naive")
        s2 = get_foundation_signal("naive")
        assert s1 is s2

    def test_forecast_dict_keys(self):
        sig = FoundationModelSignal("naive")
        result = sig.forecast(list(range(50, 100)), horizon=3)
        for key in ["model", "direction", "forecast_median", "forecast_q10", "forecast_q90"]:
            assert key in result
        # Validate full schema
        ForecastResult(**result)