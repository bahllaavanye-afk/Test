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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.ml.models.base_model import AbstractModel, EvalMetrics


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

    def train_epoch(self, loader, optimizer, criterion=None) -> dict:
        """Standard epoch training — used when loader yields (states, actions, rewards)."""
        self.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            states, actions, rewards = batch[0], batch[1], batch[2]
            dones = [False] * len(rewards)
            optimizer.zero_grad()
            result = self.actor_critic_loss(states, actions, rewards, dones)
            result["loss"].backward()
            nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            optimizer.step()
            total_loss += result["loss"].item()
            steps += 1
        return {"loss": total_loss / max(steps, 1)}

    def evaluate(self, loader) -> EvalMetrics:
        """
        Evaluate on a DataLoader that yields (x, y) pairs where
        y is the ground-truth action (0=buy, 1=hold, 2=sell).
        """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in loader:
                action_probs, _ = self.forward(X)
                preds = action_probs.argmax(dim=-1)
                correct += (preds == y.long()).sum().item()
                total += len(y)
        accuracy = correct / max(total, 1)
        return EvalMetrics(accuracy=accuracy, auc=0.5, sharpe=0.0)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str, metadata: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        init_kwargs = {
            "n_features": self.n_features,
            "hidden_size": self.hidden_size,
            "n_actions": self.n_actions,
        }
        full_meta = {"init_kwargs": init_kwargs, **(metadata or {})}
        torch.save(
            {
                "state_dict": self.state_dict(),
                "model_type": self.model_type,
                "metadata": full_meta,
            },
            path,
        )
        meta_path = Path(path).with_suffix(".json")
        meta_path.write_text(json.dumps(full_meta, default=str, indent=2))

    @classmethod
    def load(cls, path: str) -> "A3CLSTMAgent":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        init_kwargs = checkpoint.get("metadata", {}).get("init_kwargs", {})
        model = cls(**init_kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        return model
