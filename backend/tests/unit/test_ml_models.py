"""Unit tests for TFT, LightGBM, and Foundation model."""
import numpy as np
import pytest

pytest.importorskip("torch")  # skip this module when the optional [ml] extra (torch) isn't installed
import pytest as _pt
torch = _pt.importorskip("torch")
from app.ml.models.transformer import TFTModel
from app.ml.models.foundation_model import FoundationModelSignal, get_foundation_signal


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
        assert result["direction"] in (-1, 1)
        assert 0 <= result["confidence"] <= 1
        assert len(result["forecast_median"]) == 5

    def test_short_prices_returns_zero_direction(self):
        sig = FoundationModelSignal("naive")
        result = sig.forecast([100, 101, 102], horizon=3)
        assert result["direction"] == 0

    def test_get_foundation_signal_singleton(self):
        s1 = get_foundation_signal("naive")
        s2 = get_foundation_signal("naive")
        assert s1 is s2

    def test_forecast_dict_keys(self):
        sig = FoundationModelSignal("naive")
        result = sig.forecast(list(range(50, 100)), horizon=3)
        assert "model" in result
        assert "direction" in result
        assert "forecast_median" in result
        assert "forecast_q10" in result
        assert "forecast_q90" in result
