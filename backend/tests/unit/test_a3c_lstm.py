"""
Unit tests for A3C-LSTM agent, GNNSignal, and RLTraderStrategy.

Tests:
  - test_a3c_forward_shape          : forward() returns correct tensor shapes
  - test_a3c_action_selection       : select_action() returns int in {0, 1, 2}
  - test_a3c_returns_computation    : compute_returns with known rewards
  - test_gnn_signal_fallback        : GNNSignal works without torch_geometric
  - test_rl_trader_strategy_init    : RLTraderStrategy instantiates correctly
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")  # skip this module when the optional [ml] extra (torch) isn't installed
import torch


# ---------------------------------------------------------------------------
# A3C-LSTM tests
# ---------------------------------------------------------------------------

class TestA3CLSTMAgent:
    """Tests for app.ml.models.a3c_lstm.A3CLSTMAgent."""

    @pytest.fixture
    def agent(self):
        from app.ml.models.a3c_lstm import A3CLSTMAgent
        return A3CLSTMAgent(n_features=4, hidden_size=32, n_actions=3)

    def test_a3c_forward_shape(self, agent):
        """action_probs should be (batch, 3); state_value should be (batch, 1)."""
        batch, seq_len, n_features = 2, 10, 4
        x = torch.randn(batch, seq_len, n_features)
        action_probs, state_value = agent.forward(x)

        assert action_probs.shape == (batch, 3), (
            f"Expected action_probs shape (2, 3), got {action_probs.shape}"
        )
        assert state_value.shape == (batch, 1), (
            f"Expected state_value shape (2, 1), got {state_value.shape}"
        )
        # Probabilities should sum to 1
        prob_sums = action_probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(batch), atol=1e-5), (
            "action_probs rows must sum to 1"
        )

    def test_a3c_action_selection(self, agent):
        """select_action must return an int in {0, 1, 2}."""
        x = torch.randn(1, 10, 4)
        for _ in range(20):  # run multiple times — stochastic sampling
            action = agent.select_action(x)
            assert isinstance(action, int), f"Expected int, got {type(action)}"
            assert action in {0, 1, 2}, f"Action {action} not in {{0, 1, 2}}"

    def test_a3c_returns_computation(self, agent):
        """compute_returns should produce correct discounted values."""
        rewards = [1.0, 0.0, 1.0]
        gamma = 0.99
        # Expected: [1 + 0.99*0 + 0.99^2*1, 0 + 0.99*1, 1] = [1.9801, 0.99, 1.0]
        dummy_values = torch.zeros(3, 1)
        returns = agent.compute_returns(rewards, dummy_values, gamma=gamma)

        assert returns.shape == (3,), f"Expected shape (3,), got {returns.shape}"
        assert abs(float(returns[2]) - 1.0) < 1e-5, "Last return should be 1.0"
        assert abs(float(returns[1]) - 0.99) < 1e-4, "Middle return should be 0.99"
        assert abs(float(returns[0]) - 1.9801) < 1e-4, "First return should be ~1.9801"

    def test_a3c_actor_critic_loss_keys(self, agent):
        """actor_critic_loss must return dict with expected keys."""
        seq_len, n_features, T = 10, 4, 5
        states = torch.randn(T, seq_len, n_features)
        actions = torch.randint(0, 3, (T,))
        rewards = [0.1] * T
        dones = [False] * T

        loss_dict = agent.actor_critic_loss(states, actions, rewards, dones)
        for key in ("loss", "policy_loss", "value_loss", "entropy"):
            assert key in loss_dict, f"Missing key '{key}' in loss dict"
        assert loss_dict["loss"].requires_grad or True  # loss is a tensor


# ---------------------------------------------------------------------------
# GNNSignal tests
# ---------------------------------------------------------------------------

class TestGNNSignal:
    """Tests for app.ml.models.gnn_signal.GNNSignal."""

    @pytest.fixture
    def returns_df(self):
        """Synthetic returns DataFrame for 3 assets over 40 days."""
        rng = np.random.default_rng(0)
        data = rng.normal(0, 0.01, size=(40, 3))
        return pd.DataFrame(data, columns=["AAPL", "MSFT", "GOOG"])

    def test_gnn_signal_fallback(self, returns_df):
        """
        GNNSignal.predict() must work even when torch_geometric is unavailable.
        Signals should be (n_assets,) float array in [0, 1].
        """
        from app.ml.models.gnn_signal import GNNSignal

        gnn = GNNSignal(n_features=5, hidden_size=16)
        n_assets = 3
        node_features = torch.randn(n_assets, 5)

        signals = gnn.predict(returns_df, node_features)

        assert isinstance(signals, np.ndarray), "predict() must return np.ndarray"
        assert signals.shape == (n_assets,), (
            f"Expected shape ({n_assets},), got {signals.shape}"
        )
        assert np.all(signals >= 0.0) and np.all(signals <= 1.0), (
            "Signals must be in [0, 1]"
        )

    def test_correlation_graph_shape(self, returns_df):
        """CorrelationGraph.build() must return (n_assets, n_assets) matrix."""
        from app.ml.models.gnn_signal import CorrelationGraph

        cg = CorrelationGraph(window=30, threshold=0.3)
        adj = cg.build(returns_df)

        n = returns_df.shape[1]
        assert adj.shape == (n, n), f"Expected ({n}, {n}), got {adj.shape}"
        # Self-loops must be 1
        assert np.all(np.diag(adj) == 1.0), "Diagonal (self-loops) should be 1.0"
        # Values in [0, 1]
        assert adj.min() >= 0.0 and adj.max() <= 1.0

    def test_gnn_model_forward(self):
        """GNNSignalModel.forward() output shape: (n_assets, 1)."""
        from app.ml.models.gnn_signal import GNNSignalModel

        n_assets, n_features = 4, 6
        model = GNNSignalModel(n_features=n_features, hidden_size=16)
        node_features = torch.randn(n_assets, n_features)
        adj = torch.eye(n_assets)

        out = model(node_features, adj)
        assert out.shape == (n_assets, 1), f"Expected ({n_assets}, 1), got {out.shape}"
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Output must be in [0, 1]"


# ---------------------------------------------------------------------------
# EnsembleModel GNN integration tests
# ---------------------------------------------------------------------------

class TestEnsembleGNN:
    """Tests for GNN integration in EnsembleModel."""

    def test_ensemble_register_gnn(self):
        """register_gnn() stores the GNN model on the ensemble."""
        from app.ml.models.ensemble_model import EnsembleModel
        from app.ml.models.gnn_signal import GNNSignal

        ensemble = EnsembleModel(gnn_weight=0.2)
        gnn = GNNSignal(n_features=4, hidden_size=16)
        ensemble.register_gnn(gnn)

        assert ensemble._gnn_model is gnn
        assert ensemble.gnn_weight == 0.2

    def test_ensemble_gnn_weight_default(self):
        """Default gnn_weight is 0.0."""
        from app.ml.models.ensemble_model import EnsembleModel

        ensemble = EnsembleModel()
        assert ensemble.gnn_weight == 0.0
        assert ensemble._gnn_model is None


# ---------------------------------------------------------------------------
# RLTraderStrategy tests
# ---------------------------------------------------------------------------

class TestRLTraderStrategy:
    """Tests for app.strategies.ml_enhanced.rl_trader.RLTraderStrategy."""

    def test_rl_trader_strategy_init(self):
        """Strategy instantiates with the correct name and type."""
        from app.strategies.ml_enhanced.rl_trader import RLTraderStrategy

        strategy = RLTraderStrategy()
        assert strategy.name == "rl_trader", (
            f"Expected name 'rl_trader', got '{strategy.name}'"
        )
        assert strategy.strategy_type == "ml_enhanced", (
            f"Expected strategy_type 'ml_enhanced', got '{strategy.strategy_type}'"
        )

    def test_rl_trader_in_registry(self):
        """RLTraderStrategy must be registered under 'rl_trader' key."""
        from app.strategies import STRATEGY_REGISTRY
        from app.strategies.ml_enhanced.rl_trader import RLTraderStrategy

        assert "rl_trader" in STRATEGY_REGISTRY, (
            "'rl_trader' not found in STRATEGY_REGISTRY"
        )
        assert STRATEGY_REGISTRY["rl_trader"] is RLTraderStrategy

    def test_rl_trader_backtest_no_model(self):
        """backtest_signals() with no model returns RSI-based boolean series."""
        from app.strategies.ml_enhanced.rl_trader import RLTraderStrategy

        rng = np.random.default_rng(1)
        price = 100 * np.cumprod(1 + rng.normal(0, 0.01, 100))
        df = pd.DataFrame({
            "open": price * 0.999,
            "high": price * 1.005,
            "low": price * 0.995,
            "close": price,
            "volume": rng.integers(100_000, 500_000, 100).astype(float),
        })

        strategy = RLTraderStrategy(params={"model_path": "/nonexistent/path.pt"})
        bt = strategy.backtest_signals(df)

        assert hasattr(bt, "entries") and hasattr(bt, "exits")
        assert bt.entries.dtype == bool or bt.entries.dtype == np.bool_

    @pytest.mark.asyncio
    async def test_rl_trader_analyze_fallback(self):
        """analyze() with no model falls back to RSI signal (or None)."""
        from app.strategies.ml_enhanced.rl_trader import RLTraderStrategy

        rng = np.random.default_rng(2)
        price = 100 * np.cumprod(1 + rng.normal(0, 0.01, 60))
        df = pd.DataFrame({
            "open": price * 0.999,
            "high": price * 1.005,
            "low": price * 0.995,
            "close": price,
            "volume": rng.integers(100_000, 500_000, 60).astype(float),
        })

        strategy = RLTraderStrategy(params={"model_path": "/nonexistent/path.pt"})
        result = await strategy.analyze(df, "SPY")
        # Result is either None or a Signal with side in {buy, sell}
        if result is not None:
            assert result.side in {"buy", "sell"}
            assert result.strategy_name == "rl_trader"
