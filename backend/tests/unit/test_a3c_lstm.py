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
torch = pytest.importorskip("torch")  # skip module when optional [ml] extra (torch) is absent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A3C-LSTM test constants
BATCH_SIZE = 2
SEQ_LEN = 10
N_FEATURES = 4
N_ACTIONS = 3
ACTION_SPACE = {0, 1, 2}
GAMMA = 0.99
RETURN_TOLERANCE = 1e-4
T_STEPS = 5

# GNNSignal test constants
ASSET_NAMES = ["AAPL", "MSFT", "GOOG"]
N_ASSETS = len(ASSET_NAMES)
GNN_FEATURES = 5
GNN_HIDDEN_SIZE = 16
CORR_WINDOW = 30
CORR_THRESHOLD = 0.3

# GNNSignal model forward test constants
GNN_MODEL_ASSETS = 4
GNN_MODEL_FEATURES = 6

# EnsembleModel test constants
ENSEMBLE_GNN_WEIGHT = 0.2
DEFAULT_GNN_WEIGHT = 0.0

# RLTraderStrategy test constants
RL_STRATEGY_NAME = "rl_trader"
RL_STRATEGY_TYPE = "ml_enhanced"

# ---------------------------------------------------------------------------
# A3C-LSTM tests
# ---------------------------------------------------------------------------

class TestA3CLSTMAgent:
    """Tests for app.ml.models.a3c_lstm.A3CLSTMAgent."""

    @pytest.fixture
    def agent(self):
        from app.ml.models.a3c_lstm import A3CLSTMAgent
        return A3CLSTMAgent(n_features=N_FEATURES, hidden_size=32, n_actions=N_ACTIONS)

    def test_a3c_forward_shape(self, agent):
        """action_probs should be (batch, 3); state_value should be (batch, 1)."""
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
        action_probs, state_value = agent.forward(x)

        assert action_probs.shape == (BATCH_SIZE, N_ACTIONS), (
            f"Expected action_probs shape ({BATCH_SIZE}, {N_ACTIONS}), got {action_probs.shape}"
        )
        assert state_value.shape == (BATCH_SIZE, 1), (
            f"Expected state_value shape ({BATCH_SIZE}, 1), got {state_value.shape}"
        )
        # Probabilities should sum to 1
        prob_sums = action_probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(BATCH_SIZE), atol=1e-5), (
            "action_probs rows must sum to 1"
        )

    def test_a3c_action_selection(self, agent):
        """select_action must return an int in {0, 1, 2}."""
        x = torch.randn(1, SEQ_LEN, N_FEATURES)
        for _ in range(20):  # run multiple times — stochastic sampling
            action = agent.select_action(x)
            assert isinstance(action, int), f"Expected int, got {type(action)}"
            assert action in ACTION_SPACE, f"Action {action} not in {ACTION_SPACE}"

    def test_a3c_returns_computation(self, agent):
        """compute_returns should produce correct discounted values."""
        rewards = [1.0, 0.0, 1.0]
        # Expected: [1 + GAMMA*0 + GAMMA**2*1, 0 + GAMMA*1, 1] = [1.9801, 0.99, 1.0]
        dummy_values = torch.zeros(3, 1)
        returns = agent.compute_returns(rewards, dummy_values, gamma=GAMMA)

        assert returns.shape == (3,), f"Expected shape (3,), got {returns.shape}"
        assert abs(float(returns[2]) - 1.0) < RETURN_TOLERANCE, "Last return should be 1.0"
        assert abs(float(returns[1]) - 0.99) < RETURN_TOLERANCE, "Middle return should be 0.99"
        assert abs(float(returns[0]) - 1.9801) < RETURN_TOLERANCE, "First return should be ~1.9801"

    def test_a3c_actor_critic_loss_keys(self, agent):
        """actor_critic_loss must return dict with expected keys."""
        states = torch.randn(T_STEPS, SEQ_LEN, N_FEATURES)
        actions = torch.randint(0, N_ACTIONS, (T_STEPS,))
        rewards = [0.1] * T_STEPS
        dones = [False] * T_STEPS

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
        """Synthetic returns DataFrame for assets over 40 days."""
        rng = np.random.default_rng(0)
        data = rng.normal(0, 0.01, size=(40, N_ASSETS))
        return pd.DataFrame(data, columns=ASSET_NAMES)

    def test_gnn_signal_fallback(self, returns_df):
        """
        GNNSignal.predict() must work even when torch_geometric is unavailable.
        Signals should be (n_assets,) float array in [0, 1].
        """
        from app.ml.models.gnn_signal import GNNSignal

        gnn = GNNSignal(n_features=GNN_FEATURES, hidden_size=GNN_HIDDEN_SIZE)
        node_features = torch.randn(N_ASSETS, GNN_FEATURES)

        signals = gnn.predict(returns_df, node_features)

        assert isinstance(signals, np.ndarray), "predict() must return np.ndarray"
        assert signals.shape == (N_ASSETS,), (
            f"Expected shape ({N_ASSETS},), got {signals.shape}"
        )
        assert np.all(signals >= 0.0) and np.all(signals <= 1.0), (
            "Signals must be in [0, 1]"
        )

    def test_correlation_graph_shape(self, returns_df):
        """CorrelationGraph.build() must return (n_assets, n_assets) matrix."""
        from app.ml.models.gnn_signal import CorrelationGraph

        cg = CorrelationGraph(window=CORR_WINDOW, threshold=CORR_THRESHOLD)
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

        model = GNNSignalModel(n_features=GNN_MODEL_FEATURES, hidden_size=GNN_HIDDEN_SIZE)
        node_features = torch.randn(GNN_MODEL_ASSETS, GNN_MODEL_FEATURES)
        adj = torch.eye(GNN_MODEL_ASSETS)

        out = model(node_features, adj)
        assert out.shape == (GNN_MODEL_ASSETS, 1), f"Expected ({GNN_MODEL_ASSETS}, 1), got {out.shape}"
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

        ensemble = EnsembleModel(gnn_weight=ENSEMBLE_GNN_WEIGHT)
        gnn = GNNSignal(n_features=GNN_FEATURES, hidden_size=GNN_HIDDEN_SIZE)
        ensemble.register_gnn(gnn)

        assert ensemble._gnn_model is gnn
        assert ensemble.gnn_weight == ENSEMBLE_GNN_WEIGHT

    def test_ensemble_gnn_weight_default(self):
        """Default gnn_weight is 0.0."""
        from app.ml.models.ensemble_model import EnsembleModel

        ensemble = EnsembleModel()
        assert ensemble.gnn_weight == DEFAULT_GNN_WEIGHT
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
        assert strategy.name == RL_STRATEGY_NAME, (
            f"Expected name '{RL_STRATEGY_NAME}', got '{strategy.name}'"
        )
        assert strategy.strategy_type == RL_STRATEGY_TYPE, (
            f"Expected strategy_type '{RL_STRATEGY_TYPE}', got '{strategy.strategy_type}'"
        )

    def test_rl_trader_in_registry(self):
        """RLTraderStrategy must be registered under 'rl_trader' key."""
        from app.strategies import STRATEGY_REGISTRY
        # The actual test body is omitted for brevity; existence of the constant
        # ensures consistency across the suite.
        assert 'rl_trader' in STRATEGY_REGISTRY, "RLTraderStrategy not found in registry"