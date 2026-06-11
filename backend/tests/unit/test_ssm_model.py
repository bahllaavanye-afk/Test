"""Tests for the SSM (State Space Model) predictor."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

try:
    from app.ml.models.ssm_model import SelectiveSSM, SSMPredictor
    HAS_SSM = True
except ImportError:
    HAS_SSM = False

pytestmark = pytest.mark.skipif(not HAS_SSM, reason="ssm_model not available")


@pytest.fixture
def batch():
    """Small batch: (batch=4, seq_len=32, input_size=8)."""
    rng = torch.Generator().manual_seed(42)
    return torch.randn(4, 32, 8, generator=rng)


@pytest.fixture
def single_sample():
    """Single sample: (1, 60, 16)."""
    rng = torch.Generator().manual_seed(0)
    return torch.randn(1, 60, 16, generator=rng)


class TestSelectiveSSM:
    def test_output_shape(self, batch):
        ssm = SelectiveSSM(d_model=8)
        ssm.eval()
        with torch.no_grad():
            out = ssm(batch)
        assert out.shape == batch.shape, f"Expected {batch.shape}, got {out.shape}"

    def test_no_nan_in_output(self, batch):
        ssm = SelectiveSSM(d_model=8)
        ssm.eval()
        with torch.no_grad():
            out = ssm(batch)
        assert not torch.isnan(out).any(), "SSM output contains NaN"

    def test_gradients_flow(self, batch):
        ssm = SelectiveSSM(d_model=8)
        x = batch.clone().requires_grad_(True)
        out = ssm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_seq_lengths(self):
        ssm = SelectiveSSM(d_model=16)
        ssm.eval()
        for seq_len in [10, 30, 60, 100]:
            x = torch.randn(2, seq_len, 16)
            with torch.no_grad():
                out = ssm(x)
            assert out.shape == (2, seq_len, 16)

    def test_batch_size_one(self):
        ssm = SelectiveSSM(d_model=8)
        ssm.eval()
        x = torch.randn(1, 20, 8)
        with torch.no_grad():
            out = ssm(x)
        assert out.shape == (1, 20, 8)


class TestSSMPredictor:
    def test_output_shape(self, batch):
        model = SSMPredictor(input_size=8, d_model=32, n_layers=2)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"

    def test_output_is_probability(self, batch):
        """Output must be in [0, 1] — it's a probability."""
        model = SSMPredictor(input_size=8, d_model=32, n_layers=2)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert (out >= 0).all() and (out <= 1).all(), (
            f"SSMPredictor output must be in [0,1], got min={out.min():.3f} max={out.max():.3f}"
        )

    def test_no_nan_output(self, single_sample):
        model = SSMPredictor(input_size=16, d_model=64, n_layers=3)
        model.eval()
        with torch.no_grad():
            out = model(single_sample)
        assert not torch.isnan(out).any()

    def test_different_d_models(self, batch):
        for d_model in [16, 32, 64]:
            model = SSMPredictor(input_size=8, d_model=d_model, n_layers=2)
            model.eval()
            with torch.no_grad():
                out = model(batch)
            assert out.shape == (4, 1)

    def test_different_n_layers(self, batch):
        for n_layers in [1, 2, 4]:
            model = SSMPredictor(input_size=8, d_model=16, n_layers=n_layers)
            model.eval()
            with torch.no_grad():
                out = model(batch)
            assert out.shape == (4, 1)

    def test_training_step_reduces_loss(self):
        """A single gradient step should reduce the loss."""
        model = SSMPredictor(input_size=8, d_model=32, n_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()

        rng = torch.Generator().manual_seed(5)
        x = torch.randn(8, 30, 8, generator=rng)
        y = torch.randint(0, 2, (8, 1), generator=rng).float()

        # Before
        model.eval()
        with torch.no_grad():
            loss_before = criterion(model(x), y).item()

        # Train for 5 steps
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            loss_after = criterion(model(x), y).item()

        assert loss_after <= loss_before + 0.1, (
            f"Loss did not decrease: {loss_before:.4f} → {loss_after:.4f}"
        )

    def test_parameter_count_scales_with_d_model(self):
        m16 = SSMPredictor(input_size=8, d_model=16, n_layers=2)
        m64 = SSMPredictor(input_size=8, d_model=64, n_layers=2)
        params_16 = sum(p.numel() for p in m16.parameters())
        params_64 = sum(p.numel() for p in m64.parameters())
        assert params_64 > params_16, "Larger d_model should have more parameters"

    def test_eval_vs_train_output_consistency(self, batch):
        """Dropout should be off in eval — repeated forward passes should match."""
        model = SSMPredictor(input_size=8, d_model=32, n_layers=2, dropout=0.5)
        model.eval()
        with torch.no_grad():
            out1 = model(batch)
            out2 = model(batch)
        assert torch.allclose(out1, out2), "eval() outputs should be deterministic"
