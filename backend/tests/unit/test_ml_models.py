"""Unit tests for TFT, LightGBM, and Foundation model."""

import numpy as np
import pytest

# Constants for TFT model tests
BATCH_SIZE = 4
SEQ_LENGTH = 30
N_FEATURES = 10
D_MODEL = 32
N_HEADS = 2

# Constants for Foundation model signal tests
FOUNDATION_MODEL_NAME = "naive"
PRICE_START = 100.0
PRICE_STEP = 0.1
PRICE_COUNT = 50
FORECAST_HORIZON = 5
SHORT_PRICES = [100, 101, 102]
SHORT_HORIZON = 3

# Forecast result dictionary keys
KEY_MODEL = "model"
KEY_DIRECTION = "direction"
KEY_FORECAST_MEDIAN = "forecast_median"
KEY_FORECAST_Q10 = "forecast_q10"
KEY_FORECAST_Q90 = "forecast_q90"

pytest.importorskip("torch")  # skip this module when the optional [ml] extra (torch) isn't installed
torch = pytest.importorskip("torch")
from app.ml.models.transformer import TFTModel
from app.ml.models.foundation_model import FoundationModelSignal, get_foundation_signal


class TestTFTModel:
    @classmethod
    def setup_class(cls):
        """Create a single model instance for all tests to avoid repeated heavy initialization."""
        cls.model = TFTModel(
            n_features=N_FEATURES,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            seq_len=SEQ_LENGTH,
        )

    def _make_batch(self, batch=BATCH_SIZE, seq=SEQ_LENGTH, features=N_FEATURES):
        return torch.randn(batch, seq, features)

    def test_forward_shape(self):
        x = self._make_batch()
        out = self.model(x)
        assert out.shape == (BATCH_SIZE, 1)

    def test_output_bounded_0_1(self):
        x = self._make_batch()
        out = self.model(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_attention_weights_accessible(self):
        x = self._make_batch()
        self.model(x)
        weights = self.model.get_attention_weights()
        assert weights is not None

    def test_predict_proba(self):
        x = self._make_batch()
        probs = self.model.predict_proba(x)
        assert len(probs) == BATCH_SIZE
        assert all(0 <= p <= 1 for p in probs)


class TestFoundationModelSignal:
    @classmethod
    def setup_class(cls):
        """Cache a single naive foundation model signal instance."""
        cls.sig = FoundationModelSignal(FOUNDATION_MODEL_NAME)

    def test_naive_forecast(self):
        prices = [PRICE_START + i * PRICE_STEP for i in range(PRICE_COUNT)]
        result = self.sig.forecast(prices, horizon=FORECAST_HORIZON)
        assert result[KEY_DIRECTION] in (-1, 1)
        assert 0 <= result["confidence"] <= 1
        assert len(result[KEY_FORECAST_MEDIAN]) == FORECAST_HORIZON

    def test_short_prices_returns_zero_direction(self):
        result = self.sig.forecast(SHORT_PRICES, horizon=SHORT_HORIZON)
        assert result[KEY_DIRECTION] == 0

    def test_get_foundation_signal_singleton(self):
        s1 = get_foundation_signal(FOUNDATION_MODEL_NAME)
        s2 = get_foundation_signal(FOUNDATION_MODEL_NAME)
        assert s1 is s2

    def test_forecast_dict_keys(self):
        result = self.sig.forecast(list(range(50, 100)), horizon=3)
        assert KEY_MODEL in result
        assert KEY_DIRECTION in result
        assert KEY_FORECAST_MEDIAN in result
        assert KEY_FORECAST_Q10 in result
        assert KEY_FORECAST_Q90 in result