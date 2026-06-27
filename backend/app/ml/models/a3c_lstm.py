"""
A3C-LSTM Reinforcement Learning Trading Agent.

Based on: "Deep Reinforcement Learning for Quantitative Trading"
Architecture: Asynchronous Advantage Actor-Critic (A3C) with LSTM memory.

The actor network outputs: [buy, hold, sell] probabilities
The critic network outputs: state value V(s)
LSTM memory: retains market context across time steps

Used in: ml_enhanced/rl_trader.py strategy
Training: experiments/configs/ppo_execution.yaml
"""
from typing import Any, Iterable, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from app.ml.models.base_model import AbstractModel, EvalMetrics


class LSTMActor(nn.Module):
    """Policy network: maps market state sequence → action logits [buy, hold, sell]."""

    def __init__(self, n_features: int, hidden_size: int = 128, n_actions: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features) or (seq_len, n_features)
        Returns:
            logits: (batch, n_actions)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        out, _ = self.lstm(x)          # (batch, seq, hidden)
        ctx = out[:, -1, :]            # take last time step
        return self.head(ctx)          # (batch, n_actions)


class LSTMCritic(nn.Module):
    """Value network: maps market state sequence → scalar state value V(s)."""

    def __init__(self, n_features: int, hidden_size: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features) or (seq_len, n_features)
        Returns:
            value: (batch, 1)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        out, _ = self.lstm(x)
        ctx = out[:, -1, :]
        return self.head(ctx)          # (batch, 1)


class A3CLSTMAgent(AbstractModel, nn.Module):
    """
    A3C-LSTM agent used as a signal generator for the rl_trader strategy.

    Actions:
        0 = buy
        1 = hold
        2 = sell
    """

    model_type = "a3c_lstm"

    def __init__(
        self,
        n_features: int = 27,
        hidden_size: int = 128,
        n_actions: int = 3,
    ):
        nn.Module.__init__(self)
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_actions = n_actions

        self.actor = LSTMActor(n_features, hidden_size, n_actions)
        self.critic = LSTMCritic(n_features, hidden_size)

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            action_probs: (batch, n_actions)  — softmax over actions
            state_value:  (batch, 1)
        """
        logits = self.actor(x)
        action_probs = F.softmax(logits, dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, x: torch.Tensor) -> int:
        """
        Sample one action from the policy distribution.

        Args:
            x: (seq_len, n_features) or (1, seq_len, n_features)
        Returns:
            action: int in {0, 1, 2}  (buy, hold, sell)
        """
        self.eval()
        with torch.no_grad():
            action_probs, _ = self.forward(x)
            dist = torch.distributions.Categorical(probs=action_probs[0])
            return int(dist.sample().item())

    # ------------------------------------------------------------------
    # Returns computation
    # ------------------------------------------------------------------

    def compute_returns(
        self,
        rewards: List[float],
        values: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """
        Compute discounted returns with bootstrapped final value.

        Args:
            rewards: list of per-step rewards
            values:  (T, 1) critic estimates (unused for bootstrapping here)
            gamma:   discount factor
        Returns:
            returns: (T,) tensor of discounted returns
        """
        returns: List[float] = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=values.device)
        return returns_tensor

    # ------------------------------------------------------------------
    # Combined A3C loss
    # ------------------------------------------------------------------

    def actor_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: List[float],
        dones: List[bool],
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ) -> dict:
        """
        Compute actor-critic loss for one trajectory.

        Args:
            states:  (T, seq_len, n_features)
            actions: (T,) int tensor
            rewards: list[float] length T
            dones:   list[bool]  length T (unused in current formulation)
        Returns:
            dict with keys: loss, policy_loss, value_loss, entropy
        """
        action_probs, state_values = self.forward(states)  # (T, n_actions), (T, 1)
        state_values = state_values.squeeze(-1)            # (T,)

        returns = self.compute_returns(rewards, state_values.detach(), gamma)
        advantages = returns - state_values.detach()

        # Policy (actor) loss
        dist = torch.distributions.Categorical(probs=action_probs)
        log_probs = dist.log_prob(actions.long())
        policy_loss = -(log_probs * advantages).mean()

        # Value (critic) loss
        value_loss = F.mse_loss(state_values, returns)

        # Entropy bonus (encourages exploration)
        entropy = dist.entropy().mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
        }

    # ------------------------------------------------------------------
    # AbstractModel interface
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        loader: Iterable[Tuple[torch.Tensor, torch.Tensor, List[float]]],
        optimizer: torch.optim.Optimizer,
        criterion: Any = None,
    ) -> dict:
        """
        Run a single training epoch over the provided data loader.

        Each batch is expected to be a tuple (states, actions, rewards).
        Optional ``dones`` can be supplied as a fourth element; if omitted,
        a list of ``False`` values is used.

        Returns a dictionary with average loss metrics for the epoch.
        """
        self.train()
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        steps = 0

        for batch in loader:
            # Unpack batch; support both 3‑tuple and 4‑tuple formats
            if len(batch) == 3:
                states, actions, rewards = batch
                dones = [False] * len(rewards)
            else:
                states, actions, rewards, dones = batch

            optimizer.zero_grad()
            loss_dict = self.actor_critic_loss(
                states,
                actions,
                rewards,
                dones,
            )
            loss_dict["loss"].backward()
            optimizer.step()

            total_loss += loss_dict["loss"].item()
            total_policy += loss_dict["policy_loss"].item()
            total_value += loss_dict["value_loss"].item()
            total_entropy += loss_dict["entropy"].item()
            steps += 1

        if steps == 0:
            raise ValueError("Data loader yielded no batches during training epoch.")

        return {
            "avg_loss": total_loss / steps,
            "avg_policy_loss": total_policy / steps,
            "avg_value_loss": total_value / steps,
            "avg_entropy": total_entropy / steps,
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return action probabilities for given input states.
        """
        self.eval()
        with torch.no_grad():
            action_probs, _ = self.forward(x)
        return action_probs

    def evaluate(self, loader: Iterable[Any]) -> EvalMetrics:
        """
        Evaluate the model on a validation loader and return metrics.
        This implementation computes average loss using the same loss
        function as training; more sophisticated metrics can be added
        by downstream code.
        """
        self.eval()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            if len(batch) == 3:
                states, actions, rewards = batch
                dones = [False] * len(rewards)
            else:
                states, actions, rewards, dones = batch

            loss_dict = self.actor_critic_loss(
                states,
                actions,
                rewards,
                dones,
            )
            total_loss += loss_dict["loss"].item()
            steps += 1

        avg_loss = total_loss / steps if steps > 0 else float("nan")
        return EvalMetrics(loss=avg_loss)