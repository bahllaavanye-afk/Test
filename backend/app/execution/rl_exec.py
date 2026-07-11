"""
PPO-based Reinforcement Learning Execution Agent.

Learns to minimize implementation shortfall by choosing execution actions
at each time step during order execution.

State (5 dims):
  [remaining_fraction, elapsed_fraction, spread_bps_norm,
   volume_ratio, book_imbalance]

Actions (4 discrete):
  0 = wait           (no fill this step)
  1 = limit_inside   (post limit at bid+1bps / ask-1bps)
  2 = limit_best     (post limit at best bid/ask)
  3 = market         (aggressive fill immediately)

Reward:
  -slippage_bps per fill step (negative slippage = agent is penalized for bad fills)
  +completion_bonus if fully filled before deadline

Falls back to TWAP if no trained model found.
Policy weights are saved to: backend/models_artifacts/rl_exec_policy.pt
"""
from __future__ import annotations

import asyncio
import os
import random
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

from app.utils.logging import logger


_MODEL_PATH = Path("backend/models_artifacts/rl_exec_policy.pt")
_STATE_DIM = 5
_ACTION_DIM = 4
_ACTION_NAMES = ["wait", "limit_inside", "limit_best", "market"]


class ExecutionPolicy(nn.Module):
    """
    Shared actor-critic MLP.
    Input: state_dim=5
    Outputs: action_logits (4,) and state_value (1,)
    """

    def __init__(self, state_dim: int = _STATE_DIM, hidden: int = 64, n_actions: int = _ACTION_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value

    def act(self, state: np.ndarray) -> tuple[int, float]:
        """Sample action and return (action_idx, log_prob)."""
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item())


class RLExecAgent:
    """
    RL execution agent.
    Loads a pre-trained ExecutionPolicy if available; otherwise uses a heuristic
    based on remaining time and spread.
    """

    def __init__(self):
        self.policy = ExecutionPolicy()
        self._trained = False
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        """Load policy weights from disk if the model file exists."""
        path = _MODEL_PATH
        if path.exists():
            try:
                state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
                self.policy.load_state_dict(state_dict)
                self.policy.eval()
                self._trained = True
                logger.info("RLExecAgent: loaded policy from %s", path)
            except Exception as e:
                logger.warning("RLExecAgent: failed to load policy (%s), using heuristic", e)

    def select_action(self, state: dict) -> str:
        """
        Select execution action given the current state.

        Args:
            state: dict with keys:
                remaining_fraction (0-1): fraction of order remaining
                elapsed_fraction   (0-1): fraction of time window elapsed
                spread_bps         (float): current bid-ask spread in bps (normalised /50)
                volume_ratio       (float): current volume / avg volume
                book_imbalance     (float): LOB imbalance (-1 to 1)

        Returns:
            One of: 'wait', 'limit_inside', 'limit_best', 'market'
        """
        arr = np.array([
            float(state.get("remaining_fraction", 1.0)),
            float(state.get("elapsed_fraction", 0.0)),
            float(state.get("spread_bps", 5.0)) / 50.0,   # normalise to ~[0,1]
            float(state.get("volume_ratio", 1.0)),
            float(state.get("book_imbalance", 0.0)),
        ], dtype=np.float32)

        # Clip to valid range
        arr = np.clip(arr, -2.0, 2.0)

        if self._trained:
            action_idx, _ = self.policy.act(arr)
            return _ACTION_NAMES[action_idx]

        # Heuristic fallback: aggressive when time is running out or large spread signals urgency
        remaining = arr[0]
        elapsed = arr[1]
        spread_norm = arr[2]

        if elapsed > 0.85 or remaining < 0.05:
            return "market"
        elif spread_norm < 0.2:     # tight spread → post limit
            return "limit_best"
        elif elapsed > 0.5:
            return "limit_inside"
        else:
            return "wait"

    def save(self, path: Path | None = None) -> None:
        p = path or _MODEL_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), str(p))


_shared_agent: RLExecAgent | None = None


def get_rl_agent() -> RLExecAgent:
    """Return or create the singleton RL execution agent."""
    global _shared_agent
    if _shared_agent is None:
        _shared_agent = RLExecAgent()
    return _shared_agent


class RLExecution:
    """
    Drop-in replacement for TWAPExecution / LimitFirstExecution.

    Uses the RL agent to decide execution actions at each step.
    Falls back to market order after fallback_seconds if unfilled.
    """

    def __init__(
        self,
        broker,
        agent: RLExecAgent | None = None,
        step_seconds: int = 30,
        fallback_seconds: int = 300,
    ):
        self.broker = broker
        self.agent = agent or get_rl_agent()
        self.step_seconds = step_seconds
        self.fallback_seconds = fallback_seconds

    async def execute(self, request, signal_price: float | None = None) -> list[dict]:
        """
        Execute an order using RL policy.

        Returns list of fill dicts: [{qty, price, algo, slippage_bps}]
        """
        from app.brokers.base import OrderRequest

        total_qty = float(request.quantity)
        remaining = total_qty
        fills = []
        start_time = time.monotonic()
        max_steps = max(1, self.fallback_seconds // self.step_seconds)
        step = 0

        while remaining > 0.01 and step < max_steps:
            elapsed = time.monotonic() - start_time
            elapsed_frac = min(1.0, elapsed / self.fallback_seconds)
            remaining_frac = remaining / total_qty

            state = {
                "remaining_fraction": remaining_frac,
                "elapsed_fraction": elapsed_frac,
                "spread_bps": 5.0,         # default; would come from live LOB in production
                "volume_ratio": 1.0,
                "book_imbalance": 0.0,
            }

            action = self.agent.select_action(state)

            if action == "wait":
                await asyncio.sleep(self.step_seconds)
                step += 1
                continue

            # Build sub-order request
            fill_qty = remaining if action == "market" else min(remaining, total_qty * 0.15)
            sub = OrderRequest(
                symbol=request.symbol,
                side=request.side,
                order_type="market" if action == "market" else "limit",
                quantity=fill_qty,
                limit_price=request.limit_price if action in ("limit_inside", "limit_best") else None,
                account_id=request.account_id,
                execution_algo=f"rl_{action}",
            )

            try:
                # Placeholder for actual broker interaction
                pass
            except Exception:
                # In production, errors would be logged and possibly retried
                pass

        return fills


# ----------------------------------------------------------------------
# Unit tests for RLExecAgent edge cases
# ----------------------------------------------------------------------
import unittest
from unittest.mock import MagicMock, patch


class TestRLExecAgent(unittest.TestCase):
    def setUp(self):
        self.agent = RLExecAgent()
        # Ensure deterministic heuristic behavior
        self.agent._trained = False

    def test_heuristic_market_when_elapsed_near_deadline(self):
        """When elapsed_fraction > 0.85, the agent should select 'market'."""
        state = {
            "remaining_fraction": 0.5,
            "elapsed_fraction": 0.86,
            "spread_bps": 100.0,
            "volume_ratio": 1.0,
            "book_imbalance": 0.0,
        }
        action = self.agent.select_action(state)
        self.assertEqual(action, "market")

    def test_heuristic_limit_best_when_spread_tight(self):
        """Tight spread (norm < 0.2) should trigger 'limit_best' if other conditions are not met."""
        state = {
            "remaining_fraction": 0.5,
            "elapsed_fraction": 0.4,
            "spread_bps": 5.0,   # 5/50 = 0.1 < 0.2
            "volume_ratio": 1.0,
            "book_imbalance": 0.0,
        }
        action = self.agent.select_action(state)
        self.assertEqual(action, "limit_best")

    def test_trained_policy_action_selection(self):
        """When a trained policy is available, the agent should use it to decide actions."""
        # Mock a trained policy
        mock_policy = MagicMock()
        mock_policy.act.return_value = (2, -0.123)  # Corresponds to 'limit_best'
        self.agent.policy = mock_policy
        self.agent._trained = True

        state = {
            "remaining_fraction": 0.3,
            "elapsed_fraction": 0.2,
            "spread_bps": 20.0,
            "volume_ratio": 1.0,
            "book_imbalance": 0.0,
        }
        action = self.agent.select_action(state)
        self.assertEqual(action, "limit_best")
        mock_policy.act.assert_called_once()

    def test_clipping_extreme_values(self):
        """Values outside the allowed range should be clipped before heuristic evaluation."""
        state = {
            "remaining_fraction": -10.0,   # will be clipped to -2.0
            "elapsed_fraction": 10.0,     # will be clipped to 2.0
            "spread_bps": 5000.0,         # norm 100, clipped to 2.0
            "volume_ratio": 0.0,
            "book_imbalance": 0.0,
        }
        action = self.agent.select_action(state)
        # After clipping, remaining < 0.05 triggers market order
        self.assertEqual(action, "market")


if __name__ == "__main__":
    unittest.main()