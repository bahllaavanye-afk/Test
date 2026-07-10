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
import json
import logging
import time
from pathlib import Path

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    F = None      # type: ignore[assignment]

from app.ml.models.base_model import AbstractModel, EvalMetrics

logger = logging.getLogger(__name__)

class LSTMActor(nn.Module):
    """Policy network: maps market state sequence → action logits [buy, hold, sell]."""

    def __init__(self, n_features: int, hidden_size: int = 128, n_actions: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
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
        self.hidden_size = hidden_size
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

        # Monitoring counters
        self.signal_count = 0

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            action_probs: (batch, n_actions)  — softmax over actions
            state_value:  (batch, 1)
        """
        start = time.perf_counter()
        logits = self.actor(x)
        action_probs = F.softmax(logits, dim=-1)
        state_value = self.critic(x)
        elapsed = time.perf_counter() - start
        logger.info(
            "forward_pass",
            extra={"execution_time_ms": elapsed * 1000, "batch_size": x.shape[0]}
        )
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
            action = int(dist.sample().item())
            self.signal_count += 1
            logger.info(
                "signal_generated",
                extra={
                    "signal_id": self.signal_count,
                    "action": action,
                    "timestamp": time.time()
                }
            )
            return action

    # ------------------------------------------------------------------
    # Returns computation
    # ------------------------------------------------------------------

    def compute_returns(
        self,
        rewards: list[float],
        values: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """
        Compute discounted returns with bootstrapped final value.

        Args:
            rewards: list of per-step rewards
            values:  (T, 1) critic estimates
            gamma:   discount factor
        Returns:
            returns: (T,) tensor of discounted returns
        """
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # Monitoring: log cumulative return (proxy for P&L) and count of signals
        cumulative_return = returns_tensor.sum().item()
        logger.info(
            "returns_computed",
            extra={
                "cumulative_return": cumulative_return,
                "signal_count": self.signal_count,
                "num_steps": len(rewards)
            }
        )
        return returns_tensor

    # ------------------------------------------------------------------
    # Combined A3C loss
    # ------------------------------------------------------------------

    def actor_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: list[float],
        dones: list[bool],
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
            dones:   list[bool]  length T
        Returns:
            dict with keys: loss, policy_loss, value_loss, entropy
        """
        start = time.perf_counter()
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

        elapsed = time.perf_counter() - start
        logger.info(
            "actor_critic_loss",
            extra={
                "execution_time_ms": elapsed * 1000,
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item()
            }
        )

        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
        }

    # ------------------------------------------------------------------
    # AbstractModel interface
    # ------------------------------------------------------------------

    def train_epoch(self, loader, optimizer, criterion=None) -> dict:
        """Standard epoch training — used when loader yields (states, actions, rewards)."""
        self.train()
        total_loss = 0.0
        steps = 0
        epoch_start = time.perf_counter()
        for batch in loader:
            states, actions, rewards = batch[0], batch[1], batch[2]
            dones = [False] * len(rewards)
            optimizer.zero_grad()
            res
# ... (truncated for brevity)""