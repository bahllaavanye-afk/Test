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
    For production use, replace with historical Alpaca 1-min bar replay.
    """

    def __init__(self, total_steps: int = 20, spread_bps: float = 5.0):
        self.total_steps = total_steps
        self.spread_bps = spread_bps
        self.reset()

    def reset(self) -> np.ndarray:
        self.step_num = 0
        self.remaining = 1.0
        self.mid_price = 100.0 + np.random.randn() * 5
        self.arrival_price = self.mid_price
        self.volume_trend = np.random.choice([-1, 0, 1])
        return self._state()

    def _state(self) -> np.ndarray:
        volume_ratio = 1.0 + 0.3 * self.volume_trend + np.random.randn() * 0.1
        book_imbalance = np.random.uniform(-0.3, 0.3)
        return np.array([
            self.remaining,
            self.step_num / self.total_steps,
            self.spread_bps / 50.0,
            np.clip(volume_ratio, 0.1, 3.0),
            np.clip(book_imbalance, -1.0, 1.0),
        ], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Execute action. Returns (next_state, reward, done).
        Actions: 0=wait, 1=limit_inside, 2=limit_best, 3=market
        """
        # Price drift
        self.mid_price *= (1 + np.random.randn() * 0.001)
        self.step_num += 1
        done = (self.step_num >= self.total_steps) or (self.remaining < 0.01)

        if action == 0:  # wait
            reward = 0.0
        elif action == 3:  # market — immediate fill, higher slippage
            fill_size = self.remaining
            spread_cost = self.spread_bps / 2.0
            price_impact = fill_size * 2.0      # 2 bps per 100% of order
            slippage = spread_cost + price_impact
            reward = -slippage
            self.remaining = 0.0
            done = True
        else:  # limit orders
            fill_prob = 0.5 if action == 1 else 0.7   # limit_inside fills less often
            if np.random.random() < fill_prob:
                fill_size = min(self.remaining, 0.15)
                spread_cost = (self.spread_bps / 4.0) if action == 1 else (self.spread_bps / 3.0)
                slippage = spread_cost
                reward = -slippage
                self.remaining -= fill_size
            else:
                reward = 0.0

        # Penalty for not finishing on time
        if done and self.remaining > 0.01:
            reward -= self.remaining * 10.0   # urgency penalty

        return self._state(), reward, done


def compute_returns(rewards: list[float], gamma: float = 0.99) -> torch.Tensor:
    """Compute discounted returns."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    t = torch.tensor(returns, dtype=torch.float32)
    # Normalise
    if t.std() > 1e-8:
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
    Train ExecutionPolicy with REINFORCE + value baseline (Actor-Critic).
    Returns final average reward.
    """
    save_path = save_path or _MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    policy = ExecutionPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    env = ExecutionEnv()

    best_avg_reward = -float("inf")
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        log_probs, values, rewards = [], [], []

        for _ in range(env.total_steps + 5):
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = policy(x)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            values.append(value.squeeze())

            state, reward, done = env.step(int(action.item()))
            rewards.append(reward)

            if done:
                break

        returns = compute_returns(rewards, gamma)
        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)

        advantages = returns - values_t.detach()

        # Actor loss (policy gradient)
        actor_loss = -(log_probs_t * advantages).mean()
        # Critic loss
        critic_loss = nn.functional.mse_loss(values_t, returns)
        # Entropy bonus (encourages exploration)
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
