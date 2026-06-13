"""
Unit tests for self-supervised masked-bar pretraining.

Uses small synthetic sequences (a fixture, not production data). Verifies the
encoder state has transferable LSTM keys, transfer copies matching tensors into
an LSTMPredictor, and shape/empty guards raise rather than silently misbehave.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

torch = pytest.importorskip("torch")

from app.ml.training.pretrain import pretrain_masked, transfer_encoder_weights, PretrainResult


def _seqs(n=32, t=20, f=8, seed=1):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.normal(size=(n, t, f)), dtype=torch.float32)


class TestPretrainMasked:
    def test_returns_result_with_encoder_keys(self):
        res = pretrain_masked(_seqs(), n_features=8, hidden_size=16, num_layers=1,
                              bidirectional=False, epochs=3)
        assert isinstance(res, PretrainResult)
        assert res.n_sequences == 32
        assert res.epochs == 3
        # All exported keys are lstm.* so they map onto an LSTMPredictor.
        assert res.encoder_state
        assert all(k.startswith("lstm.") for k in res.encoder_state)

    def test_loss_is_finite(self):
        res = pretrain_masked(_seqs(), n_features=8, hidden_size=16, num_layers=1,
                              bidirectional=False, epochs=5)
        assert np.isfinite(res.final_loss)

    def test_wrong_feature_count_raises(self):
        with pytest.raises(ValueError):
            pretrain_masked(_seqs(f=8), n_features=10, epochs=1)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            pretrain_masked(torch.zeros(10, 8), n_features=8, epochs=1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pretrain_masked(torch.zeros(0, 20, 8), n_features=8, epochs=1)


class TestTransfer:
    def test_transfers_matching_tensors_into_predictor(self):
        from app.ml.models.lstm import LSTMPredictor
        predictor = LSTMPredictor(n_features=8, hidden_size=16, num_layers=1, bidirectional=False)
        res = pretrain_masked(_seqs(f=8), n_features=8, hidden_size=16, num_layers=1,
                              bidirectional=False, epochs=2)
        n = transfer_encoder_weights(res.encoder_state, predictor)
        assert n > 0  # at least the LSTM weight/bias tensors transferred

    def test_transfer_actually_changes_weights(self):
        from app.ml.models.lstm import LSTMPredictor
        predictor = LSTMPredictor(n_features=8, hidden_size=16, num_layers=1, bidirectional=False)
        before = predictor.state_dict()["lstm.weight_ih_l0"].clone()
        res = pretrain_masked(_seqs(f=8), n_features=8, hidden_size=16, num_layers=1,
                              bidirectional=False, epochs=4)
        transfer_encoder_weights(res.encoder_state, predictor)
        after = predictor.state_dict()["lstm.weight_ih_l0"]
        assert not torch.equal(before, after)

    def test_shape_mismatch_skipped_not_crash(self):
        from app.ml.models.lstm import LSTMPredictor
        # Predictor with different hidden size → no keys match → 0 transferred, no crash.
        predictor = LSTMPredictor(n_features=8, hidden_size=32, num_layers=1, bidirectional=False)
        res = pretrain_masked(_seqs(f=8), n_features=8, hidden_size=16, num_layers=1,
                              bidirectional=False, epochs=2)
        n = transfer_encoder_weights(res.encoder_state, predictor)
        assert n == 0
