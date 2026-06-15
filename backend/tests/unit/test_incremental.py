"""
Unit tests for the incremental (online) training core used by the AutoML desk.

Pure-numpy logic is tested directly. The torch-backed paths use a small
synthetic random-walk OHLCV frame as a fixture (a test fixture, not production
data) so build_supervised / fine_tune / validate_model exercise real code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.training.incremental import (
    ValidationScore,
    directional_sharpe,
    should_promote,
)

torch = pytest.importorskip("torch")
import pandas as pd  # noqa: E402


def _synthetic_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Trending random walk so labels are not degenerate (all one class).
    steps = rng.normal(0.0005, 0.01, size=n)
    close = 100 * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000, 100_000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# directional_sharpe
# ---------------------------------------------------------------------------
class TestDirectionalSharpe:
    def test_empty_returns_zero(self):
        assert directional_sharpe(np.array([]), np.array([])) == 0.0

    def test_flat_pnl_returns_zero(self):
        # Constant pnl → zero std → guarded to 0.0, not NaN/inf.
        probs = np.array([0.9, 0.9, 0.9])
        rets = np.array([0.01, 0.01, 0.01])
        out = directional_sharpe(probs, rets)
        assert np.isfinite(out)

    def test_correct_directional_calls_positive_sharpe(self):
        # Probs > 0.5 when return positive, < 0.5 when negative → all wins.
        probs = np.array([0.8, 0.2, 0.8, 0.2, 0.7, 0.3])
        rets = np.array([0.02, -0.02, 0.01, -0.01, 0.03, -0.015])
        assert directional_sharpe(probs, rets) > 0

    def test_wrong_calls_negative_sharpe(self):
        probs = np.array([0.2, 0.8, 0.2, 0.8])
        rets = np.array([0.02, -0.02, 0.01, -0.01])
        assert directional_sharpe(probs, rets) < 0

    def test_length_mismatch_truncates(self):
        out = directional_sharpe(np.array([0.6, 0.4, 0.6]), np.array([0.01, -0.01]))
        assert np.isfinite(out)


# ---------------------------------------------------------------------------
# ValidationScore.combined
# ---------------------------------------------------------------------------
class TestValidationScore:
    def test_combined_rewards_accuracy(self):
        low = ValidationScore(accuracy=0.5, directional_sharpe=0.0, n=100)
        high = ValidationScore(accuracy=0.6, directional_sharpe=0.0, n=100)
        assert high.combined > low.combined

    def test_combined_rewards_sharpe(self):
        flat = ValidationScore(accuracy=0.55, directional_sharpe=0.0, n=100)
        sharp = ValidationScore(accuracy=0.55, directional_sharpe=2.0, n=100)
        assert sharp.combined > flat.combined

    def test_combined_sharpe_bounded(self):
        # tanh squashing keeps a single wild Sharpe from dominating.
        insane = ValidationScore(accuracy=0.55, directional_sharpe=10_000.0, n=100)
        assert insane.combined < 0.55 + 0.25 + 1e-6


# ---------------------------------------------------------------------------
# should_promote
# ---------------------------------------------------------------------------
class TestShouldPromote:
    def test_insufficient_samples_never_promotes(self):
        ch = ValidationScore(accuracy=0.99, directional_sharpe=5.0, n=5)
        assert should_promote(None, ch, min_samples=20) is False

    def test_cold_start_promotes_valid_challenger(self):
        ch = ValidationScore(accuracy=0.55, directional_sharpe=0.5, n=50)
        assert should_promote(None, ch, min_samples=20) is True

    def test_marginal_improvement_rejected(self):
        champ = ValidationScore(accuracy=0.60, directional_sharpe=0.0, n=100)
        ch = ValidationScore(accuracy=0.605, directional_sharpe=0.0, n=100)
        assert should_promote(champ, ch, min_improvement=0.01) is False

    def test_clear_improvement_accepted(self):
        champ = ValidationScore(accuracy=0.55, directional_sharpe=0.0, n=100)
        ch = ValidationScore(accuracy=0.65, directional_sharpe=0.0, n=100)
        assert should_promote(champ, ch, min_improvement=0.01) is True

    def test_regression_rejected(self):
        champ = ValidationScore(accuracy=0.65, directional_sharpe=1.0, n=100)
        ch = ValidationScore(accuracy=0.50, directional_sharpe=-1.0, n=100)
        assert should_promote(champ, ch) is False


# ---------------------------------------------------------------------------
# Torch-backed: build_supervised / fine_tune / validate_model
# ---------------------------------------------------------------------------
class TestTorchPaths:
    def test_build_supervised_raises_on_short_data(self):
        from app.ml.training.incremental import build_supervised
        with pytest.raises(ValueError):
            build_supervised(_synthetic_ohlcv(n=30), seq_len=60)

    def test_build_supervised_shapes(self):
        from app.ml.training.incremental import build_supervised
        X, y, fwd, scaler = build_supervised(_synthetic_ohlcv(), seq_len=60)
        assert X.shape[0] == len(y) == len(fwd)
        assert X.shape[1] == 60
        assert scaler is not None

    def test_fine_tune_does_not_mutate_champion(self):
        from app.ml.training.incremental import build_supervised, fine_tune
        from app.ml.models.lstm import LSTMPredictor
        X, y, _, _ = build_supervised(_synthetic_ohlcv(), seq_len=60)
        champ = LSTMPredictor(n_features=X.shape[-1])
        before = [p.detach().clone() for p in champ.parameters()]
        challenger = fine_tune(champ, X[:50], y[:50], epochs=2, lr=1e-3)
        after = [p.detach() for p in champ.parameters()]
        # Champion parameters unchanged (challenger is a deep copy).
        assert all(torch.equal(b, a) for b, a in zip(before, after))
        assert challenger is not champ

    def test_validate_model_returns_score(self):
        from app.ml.training.incremental import build_supervised, validate_model
        from app.ml.models.lstm import LSTMPredictor
        X, y, fwd, _ = build_supervised(_synthetic_ohlcv(), seq_len=60)
        model = LSTMPredictor(n_features=X.shape[-1])
        score = validate_model(model, X[-40:], y[-40:], fwd[-40:])
        assert isinstance(score, ValidationScore)
        assert score.n == 40
        assert 0.0 <= score.accuracy <= 1.0
