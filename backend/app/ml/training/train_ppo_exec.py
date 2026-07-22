"""
PPO training script for the RL execution agent.

Simulates order execution using historical price data.
Trains ExecutionPolicy to minimize implementation shortfall.

Usage:
    python -m app.ml.training.train_ppo_exec [--episodes 100] [--symbol SPY]

Saves trained policy to: backend/models_artifacts/rl_exec_policy.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from app.execution.rl_exec import ExecutionPolicy, _MODEL_PATH, _STATE_DIM, _ACTION_DIM


class ExecutionEnv:
    """
    Simulated execution environment.
    Uses a simple price process: mid-price with random walk + spread.
    For production use, replace with historical Alpaca 1‑min bar replay.
    """

    def __init__(self, total_steps: int = 20, spread_bps: float = 5.0):
        self.total_steps = total_steps
        self.spread_bps = spread_bps
        self.reset()

    def reset(self) -> np.ndarray:
        self.step_num = 0
        self.remaining = 1.0  # proportion of order left to execute
        self.mid_price = 100.0 + np.random.randn() * 5
        self.arrival_price = self.mid_price
        self.volume_trend = np.random.choice([-1, 0, 1])  # -1: decreasing, 0: flat, 1: increasing
        self.book_imbalance = np.random.uniform(-0.3, 0.3)  # negative: sell pressure, positive: buy pressure
        return self._state()

    def _state(self) -> np.ndarray:
        """
        Returns a compact representation of the market state.
        Elements:
            0 - remaining proportion of the order
            1 - normalized step (0‑1)
            2 - normalized spread
            3 - volume ratio (adjusted for trend)
            4 - book imbalance (clipped)
        """
        volume_ratio = 1.0 + 0.3 * self.volume_trend + np.random.randn() * 0.1
        return np.array(
            [
                self.remaining,
                self.step_num / self.total_steps,
                self.spread_bps / 50.0,
                np.clip(volume_ratio, 0.1, 3.0),
                np.clip(self.book_imbalance, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _action_allowed(self, action: int) -> bool:
        """
        Confirmation filter for actions.
        - Wait (0) is always allowed.
        - Market (3) is only allowed when price momentum is favorable
          (mid‑price trending down for a buy order) or when urgency is high.
        - Limit_inside (1) requires positive book imbalance and upward volume trend.
        - Limit_best (2) requires either positive imbalance or upward volume trend.
        Returns True if the action passes the filter, otherwise False.
        """
        # Simple momentum estimate based on recent price drift
        price_momentum = self.mid_price - self.arrival_price

        if action == 0:
            return True
        if action == 3:
            # Allow market if price is moving down (cheaper) or we are near the horizon
            return price_momentum < 0 or self.step_num >= self.total_steps - 2
        if action == 1:
            # Inside limit needs both buy‑side signals
            return self.book_imbalance > 0.1 and self.volume_trend > 0
        if action == 2:
            # Best limit needs at least one favorable signal
            return self.book_imbalance > 0.0 or self.volume_trend > 0
        return False

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Execute an action and return (next_state, reward, done).

        Actions:
            0 – wait
            1 – limit_inside
            2 – limit_best
            3 – market
        """
        # Apply confirmation filter; fallback to wait if not allowed
        if not self._action_allowed(action):
            action = 0

        # Simulate small random drift of the mid‑price
        self.mid_price *= 1 + np.random.randn() * 0.001
        self.step_num += 1
        done = (self.step_num >= self.total_steps) or (self.remaining < 0.01)

        reward = 0.0

        if action == 0:  # wait
            reward = -0.01 * self.remaining  # small holding cost to discourage idle time
        elif action == 3:  # market – immediate fill, higher slippage
            fill_size = self.remaining
            spread_cost = self.spread_bps / 2.0
            price_impact = fill_size * 2.0  # 2 bps per 100 % of order
            slippage = spread_cost + price_impact
            reward = -slippage
            self.remaining = 0.0
            done = True
        else:  # limit orders
            # Fill probability reflects aggressiveness
            fill_prob = 0.5 if action == 1 else 0.7
            if np.random.random() < fill_prob:
                fill_size = min(self.remaining, 0.15)
                spread_cost = (self.spread_bps / 4.0) if action == 1 else (self.spread_bps / 3.0)
                slippage = spread_cost
                reward = -slippage
                self.remaining -= fill_size
            else:
                # Missed limit incurs a small opportunity cost
                reward = -0.005 * self.remaining

        # Penalty for unfinished order when episode ends
        if done and self.remaining > 0.01:
            reward -= self.remaining * 12.0  # higher urgency penalty than before

        # Early termination if remaining is negligible
        if self.remaining < 0.005:
            done = True

        return self._state(), reward, done


def compute_returns(rewards: list[float], gamma: float = 0.99) -> torch.Tensor:
    """Compute discounted returns with optional normalisation."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    t = torch.tensor(returns, dtype=torch.float32)
    if t.numel() > 1 and t.std() > 1e-8:
        t = (t - t.mean()) / (t.std() + 1e-8)
    return t


def train(
    n_episodes: int = 200,
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_coeff: float = 0.01,
    save_path: Path | None = None,
) -> float:
    """
    Train ExecutionPolicy with REINFORCE + value baseline (Actor‑Critic).
    Returns the best average reward observed.
    """
    save_path = save_path or _MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    policy = ExecutionPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    env = ExecutionEnv()

    best_avg_reward = -float("inf")
    episode_rewards: list[float] = []

    for episode in range(n_episodes):
        state = env.reset()
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []

        for _ in range(env.total_steps + 5):
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = policy(x)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze())

            state, reward, done = env.step(int(action.item()))
            rewards.append(reward)

            if done:
                break

        returns = compute_returns(rewards, gamma)
        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)

        advantages = returns - values_t.detach()

        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = nn.functional.mse_loss(values_t, returns)
        entropy_loss = -dist.entropy().mean()

        loss = actor_loss + 0.5 * critic_loss + entropy_coeff * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)

        if (episode + 1) % 20 == 0:
            avg = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1}/{n_episodes}  avg_reward={avg:.3f}", flush=True)
            if avg > best_avg_reward:
                best_avg_reward = avg
                torch.save(policy.state_dict(), str(save_path))

    print(f"Training complete. Best avg reward: {best_avg_reward:.3f}")
    print(f"Policy saved to {save_path}")
    return best_avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train(n_episodes=args.episodes, lr=args.lr)