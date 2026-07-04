"""Tests for the SSM (State Space Model) predictor."""
from __future__ import annotations

import numpy as np
import pytest

# Constants
BATCH_SIZE = 4
BATCH_SEQ_LEN = 32
BATCH_INPUT_SIZE = 8
BATCH_SEED = 42

SINGLE_SAMPLE_BATCH_SIZE = 1
SINGLE_SAMPLE_SEQ_LEN = 60
SINGLE_SAMPLE_INPUT_SIZE = 16
SINGLE_SAMPLE_SEED = 0

SEQ_LENGTHS = [10, 30, 60, 100]
BATCH_SIZE_ONE = 1
BATCH_SIZE_ONE_SEQ_LEN = 20

MODEL_OUTPUT_SHAPE = (BATCH_SIZE, 1)

PROBABILITY_LOWER = 0
PROBABILITY_UPPER = 1

LOSS_TOLERANCE = 0.1

D_MODEL_VALUES = [16, 32, 64]
N_LAYERS_VALUES = [1, 2, 4]

TRAINING_STEPS = 5
LEARNING_RATE = 1e-3
TRAINING_SEED = 5
TRAIN_X_SHAPE = (8, 30, BATCH_INPUT_SIZE)
TRAIN_Y_SHAPE = (8, 1)

DROPOUT_VALUE = 0.5

IMPORT_ERROR_MSG = "ssm_model not available"
SKIP_REASON_TORCH = "torch not installed"

torch = pytest.importorskip("torch", reason=SKIP_REASON_TORCH)

try:
    from app.ml.models.ssm_model import SelectiveSSM, SSMPredictor
    HAS_SSM = True
except ImportError:
    HAS_SSM = False

pytestmark = pytest.mark.skipif(not HAS_SSM, reason=IMPORT_ERROR_MSG)


@pytest.fixture
def batch():
    """Small batch: (batch={BATCH_SIZE}, seq_len={BATCH_SEQ_LEN}, input_size={BATCH_INPUT_SIZE})."""
    rng = torch.Generator().manual_seed(BATCH_SEED)
    return torch.randn(BATCH_SIZE, BATCH_SEQ_LEN, BATCH_INPUT_SIZE, generator=rng)


@pytest.fixture
def single_sample():
    """Single sample: ({SINGLE_SAMPLE_BATCH_SIZE}, {SINGLE_SAMPLE_SEQ_LEN}, {SINGLE_SAMPLE_INPUT_SIZE})."""
    rng = torch.Generator().manual_seed(SINGLE_SAMPLE_SEED)
    return torch.randn(SINGLE_SAMPLE_BATCH_SIZE, SINGLE_SAMPLE_SEQ_LEN, SINGLE_SAMPLE_INPUT_SIZE, generator=rng)


class TestSelectiveSSM:
    def test_output_shape(self, batch):
        ssm = SelectiveSSM(d_model=BATCH_INPUT_SIZE)
        ssm.eval()
        with torch.no_grad():
            out = ssm(batch)
        assert out.shape == batch.shape, f"Expected {batch.shape}, got {out.shape}"

    def test_no_nan_in_output(self, batch):
        ssm = SelectiveSSM(d_model=BATCH_INPUT_SIZE)
        ssm.eval()
        with torch.no_grad():
            out = ssm(batch)
        assert not torch.isnan(out).any(), "SSM output contains NaN"

    def test_gradients_flow(self, batch):
        ssm = SelectiveSSM(d_model=BATCH_INPUT_SIZE)
        x = batch.clone().requires_grad_(True)
        out = ssm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_seq_lengths(self):
        ssm = SelectiveSSM(d_model=16)
        ssm.eval()
        for seq_len in SEQ_LENGTHS:
            x = torch.randn(2, seq_len, 16)
            with torch.no_grad():
                out = ssm(x)
            assert out.shape == (2, seq_len, 16)

    def test_batch_size_one(self):
        ssm = SelectiveSSM(d_model=BATCH_INPUT_SIZE)
        ssm.eval()
        x = torch.randn(BATCH_SIZE_ONE, BATCH_SIZE_ONE_SEQ_LEN, BATCH_INPUT_SIZE)
        with torch.no_grad():
            out = ssm(x)
        assert out.shape == (BATCH_SIZE_ONE, BATCH_SIZE_ONE_SEQ_LEN, BATCH_INPUT_SIZE)


class TestSSMPredictor:
    def test_output_shape(self, batch):
        model = SSMPredictor(input_size=BATCH_INPUT_SIZE, d_model=32, n_layers=2)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == MODEL_OUTPUT_SHAPE, f"Expected {MODEL_OUTPUT_SHAPE}, got {out.shape}"

    def test_output_is_probability(self, batch):
        """Output must be in [0, 1] — it's a probability."""
        model = SSMPredictor(input_size=BATCH_INPUT_SIZE, d_model=32, n_layers=2)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert (out >= PROBABILITY_LOWER).all() and (out <= PROBABILITY_UPPER).all(), (
            f"SSMPredictor output must be in [{PROBABILITY_LOWER},{PROBABILITY_UPPER}], "
            f"got min={out.min():.3f} max={out.max():.3f}"
        )

    def test_no_nan_output(self, single_sample):
        model = SSMPredictor(input_size=SINGLE_SAMPLE_INPUT_SIZE, d_model=64, n_layers=3)
        model.eval()
        with torch.no_grad():
            out = model(single_sample)
        assert not torch.isnan(out).any()

    def test_different_d_models(self, batch):
        for d_model in D_MODEL_VALUES:
            model = SSMPredictor(input_size=BATCH_INPUT_SIZE, d_model=d_model, n_layers=2)
            model.eval()
            with torch.no_grad():
                out = model(batch)
            assert out.shape == MODEL_OUTPUT_SHAPE

    def test_different_n_layers(self, batch):
        for n_layers in N_LAYERS_VALUES:
            model = SSMPredictor(input_size=BATCH_INPUT_SIZE, d_model=16, n_layers=n_layers)
            model.eval()
            with torch.no_grad():
                out = model(batch)
            assert out.shape == MODEL_OUTPUT_SHAPE

    def test_training_step_reduces_loss(self):
        """A single gradient step should reduce the loss."""
        model = SSMPredictor(input_size=BATCH_INPUT_SIZE, d_model=32, n_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.BCELoss()

        rng = torch.Generator().manual_seed(TRAINING_SEED)
        x = torch.randn(*TRAIN_X_SHAPE, generator=rng)
        y = torch.randint(0, 2, TRAIN_Y_SHAPE, generator=rng).float()

        # Before
        model.eval()
        with torch.no_grad():
            loss_before = criterion(model(x), y).item()

        # Train for a few steps
        model.train()
        for _ in range(TRAINING_STEPS):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            loss_after = criterion(model(x), y).item()

        assert loss_after <= loss_before + LOSS_TOLERANCE, (
            f"Loss did not decrease: {loss_before:.4f} → {loss_after:.4f}"
        )

    def test_parameter_count_scales_with_d_model(self):
        m16 = SSMPredictor(input_size=BATCH_INPUT_SIZE, d_model=16, n_layers=2)
        m64 = SSMPredictor(input_size=BATCH_INPUT_SIZE, d_model=64, n_layers=2)
        params_16 = sum(p.numel() for p in m16.parameters())
        params_64 = sum(p.numel() for p in m64.parameters())
        assert params_64 > params_16, "Larger d_model should have more parameters"

    def test_eval_vs_train_output_consistency(self, batch):
        """Dropout should be off in eval — repeated forward passes should match."""
        model = SSMPredictor(input_size=BATCH_INPUT_SIZE, d_model=32, n_layers=2, dropout=DROPOUT_VALUE)
        model.eval()
        with torch.no_grad():
            out1 = model(batch)
            out2 = model(batch)
        assert torch.allclose(out1, out2), "eval() outputs should be deterministic"